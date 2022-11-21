//===--- Record.cpp - Record compiler events ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Record.h"
#include "clang-include-cleaner/Types.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang::include_cleaner {
namespace {

class PPRecorder : public PPCallbacks {
public:
  PPRecorder(RecordedPP &Recorded, const Preprocessor &PP)
      : Recorded(Recorded), PP(PP), SM(PP.getSourceManager()) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    Active = SM.isWrittenInMainFile(Loc);
  }

  void InclusionDirective(SourceLocation Hash, const Token &IncludeTok,
                          StringRef SpelledFilename, bool IsAngled,
                          CharSourceRange FilenameRange,
                          llvm::Optional<FileEntryRef> File,
                          StringRef SearchPath, StringRef RelativePath,
                          const Module *, SrcMgr::CharacteristicKind) override {
    if (!Active)
      return;

    Include I;
    I.HashLocation = Hash;
    I.Resolved = File ? &File->getFileEntry() : nullptr;
    I.Line = SM.getSpellingLineNumber(Hash);
    I.Spelled = SpelledFilename;
    Recorded.Includes.add(I);
  }

  void MacroExpands(const Token &MacroName, const MacroDefinition &MD,
                    SourceRange Range, const MacroArgs *Args) override {
    if (!Active)
      return;
    recordMacroRef(MacroName, *MD.getMacroInfo());
  }

  void MacroDefined(const Token &MacroName, const MacroDirective *MD) override {
    if (!Active)
      return;

    const auto *MI = MD->getMacroInfo();
    // The tokens of a macro definition could refer to a macro.
    // Formally this reference isn't resolved until this macro is expanded,
    // but we want to treat it as a reference anyway.
    for (const auto &Tok : MI->tokens()) {
      auto *II = Tok.getIdentifierInfo();
      // Could this token be a reference to a macro? (Not param to this macro).
      if (!II || !II->hadMacroDefinition() ||
          llvm::is_contained(MI->params(), II))
        continue;
      if (const MacroInfo *MI = PP.getMacroInfo(II))
        recordMacroRef(Tok, *MI);
    }
  }

  void MacroUndefined(const Token &MacroName, const MacroDefinition &MD,
                      const MacroDirective *) override {
    if (!Active)
      return;
    if (const auto *MI = MD.getMacroInfo())
      recordMacroRef(MacroName, *MI);
  }

private:
  void recordMacroRef(const Token &Tok, const MacroInfo &MI) {
    if (MI.isBuiltinMacro())
      return; // __FILE__ is not a reference.
    Recorded.MacroReferences.push_back(
        SymbolReference{Tok.getLocation(),
                        Macro{Tok.getIdentifierInfo(), MI.getDefinitionLoc()},
                        RefType::Explicit});
  }

  bool Active = false;
  RecordedPP &Recorded;
  const Preprocessor &PP;
  const SourceManager &SM;
};

} // namespace

// FIXME: this is a mirror of clang::clangd::parseIWYUPragma, move to libTooling
// to share the code?
static llvm::Optional<StringRef> parseIWYUPragma(const char *Text) {
  assert(strncmp(Text, "//", 2) || strncmp(Text, "/*", 2));
  constexpr llvm::StringLiteral IWYUPragma = " IWYU pragma: ";
  Text += 2; // Skip the comment start, // or /*.
  if (strncmp(Text, IWYUPragma.data(), IWYUPragma.size()))
    return llvm::None;
  Text += IWYUPragma.size();
  const char *End = Text;
  while (*End != 0 && *End != '\n')
    ++End;
  return StringRef(Text, End - Text);
}

class PragmaIncludes::RecordPragma : public PPCallbacks, public CommentHandler {
public:
  RecordPragma(const CompilerInstance &CI, PragmaIncludes *Out)
      : SM(CI.getSourceManager()), Out(Out) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    InMainFile = SM.isWrittenInMainFile(Loc);
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange /*FilenameRange*/,
                          Optional<FileEntryRef> File,
                          llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const clang::Module * /*Imported*/,
                          SrcMgr::CharacteristicKind FileKind) override {
    if (!InMainFile)
      return;
    int HashLine =
        SM.getLineNumber(SM.getMainFileID(), SM.getFileOffset(HashLoc));
    if (LastPragmaKeepInMainFileLine == HashLine)
      Out->ShouldKeep.insert(HashLine);
  }

  bool HandleComment(Preprocessor &PP, SourceRange Range) override {
    auto &SM = PP.getSourceManager();
    auto Pragma = parseIWYUPragma(SM.getCharacterData(Range.getBegin()));
    if (!Pragma)
      return false;

    if (Pragma->consume_front("private, include ")) {
      // We always insert using the spelling from the pragma.
      if (auto *FE = SM.getFileEntryForID(SM.getFileID(Range.getBegin())))
        Out->IWYUPublic.insert(
            {FE->getLastRef().getUniqueID(),
             Pragma->startswith("<") || Pragma->startswith("\"")
                 ? Pragma->str()
                 : ("\"" + *Pragma + "\"").str()});
      return false;
    }

    if (InMainFile) {
      if (!Pragma->startswith("keep"))
        return false;
      // Given:
      //
      // #include "foo.h"
      // #include "bar.h" // IWYU pragma: keep
      //
      // The order in which the callbacks will be triggered:
      //
      // 1. InclusionDirective("foo.h")
      // 2. handleCommentInMainFile("// IWYU pragma: keep")
      // 3. InclusionDirective("bar.h")
      //
      // This code stores the last location of "IWYU pragma: keep" (or export)
      // comment in the main file, so that when next InclusionDirective is
      // called, it will know that the next inclusion is behind the IWYU pragma.
      LastPragmaKeepInMainFileLine = SM.getLineNumber(
          SM.getMainFileID(), SM.getFileOffset(Range.getBegin()));
    }
    return false;
  }

private:
  bool InMainFile = false;
  const SourceManager &SM;
  PragmaIncludes *Out;
  // Track the last line "IWYU pragma: keep" was seen in the main file, 1-based.
  int LastPragmaKeepInMainFileLine = -1;
};

void PragmaIncludes::record(const CompilerInstance &CI) {
  auto Record = std::make_unique<RecordPragma>(CI, this);
  CI.getPreprocessor().addCommentHandler(Record.get());
  CI.getPreprocessor().addPPCallbacks(std::move(Record));
}

llvm::StringRef PragmaIncludes::getPublic(const FileEntry *F) const {
  auto It = IWYUPublic.find(F->getUniqueID());
  if (It == IWYUPublic.end())
    return "";
  return It->getSecond();
}

std::unique_ptr<ASTConsumer> RecordedAST::record() {
  class Recorder : public ASTConsumer {
    RecordedAST *Out;

  public:
    Recorder(RecordedAST *Out) : Out(Out) {}
    void Initialize(ASTContext &Ctx) override { Out->Ctx = &Ctx; }
    bool HandleTopLevelDecl(DeclGroupRef DG) override {
      const auto &SM = Out->Ctx->getSourceManager();
      for (Decl *D : DG) {
        if (!SM.isWrittenInMainFile(SM.getExpansionLoc(D->getLocation())))
          continue;
        // FIXME: Filter out certain Obj-C and template-related decls.
        Out->Roots.push_back(D);
      }
      return ASTConsumer::HandleTopLevelDecl(DG);
    }
  };

  return std::make_unique<Recorder>(this);
}

void RecordedPP::RecordedIncludes::add(const Include &I) {
  unsigned Index = All.size();
  All.push_back(I);
  auto BySpellingIt = BySpelling.try_emplace(I.Spelled).first;
  All.back().Spelled = BySpellingIt->first(); // Now we own the backing string.

  BySpellingIt->second.push_back(Index);
  if (I.Resolved)
    ByFile[I.Resolved].push_back(Index);
}

llvm::SmallVector<const Include *>
RecordedPP::RecordedIncludes::match(Header H) const {
  llvm::SmallVector<const Include *> Result;
  switch (H.kind()) {
  case Header::Physical:
    for (unsigned I : ByFile.lookup(H.physical()))
      Result.push_back(&All[I]);
    break;
  case Header::Standard:
    for (unsigned I : BySpelling.lookup(H.standard().name().trim("<>")))
      Result.push_back(&All[I]);
    break;
  }
  return Result;
}

std::unique_ptr<PPCallbacks> RecordedPP::record(const Preprocessor &PP) {
  return std::make_unique<PPRecorder>(*this, PP);
}

} // namespace clang::include_cleaner
