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
#include "clang/Tooling/Inclusions/HeaderAnalysis.h"

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

  void Ifdef(SourceLocation Loc, const Token &MacroNameTok,
             const MacroDefinition &MD) override {
    if (!Active)
      return;
    if (const auto *MI = MD.getMacroInfo())
      recordMacroRef(MacroNameTok, *MI, RefType::Ambiguous);
  }

  void Ifndef(SourceLocation Loc, const Token &MacroNameTok,
              const MacroDefinition &MD) override {
    if (!Active)
      return;
    if (const auto *MI = MD.getMacroInfo())
      recordMacroRef(MacroNameTok, *MI, RefType::Ambiguous);
  }

  void Elifdef(SourceLocation Loc, const Token &MacroNameTok,
               const MacroDefinition &MD) override {
    if (!Active)
      return;
    if (const auto *MI = MD.getMacroInfo())
      recordMacroRef(MacroNameTok, *MI, RefType::Ambiguous);
  }

  void Elifndef(SourceLocation Loc, const Token &MacroNameTok,
                const MacroDefinition &MD) override {
    if (!Active)
      return;
    if (const auto *MI = MD.getMacroInfo())
      recordMacroRef(MacroNameTok, *MI, RefType::Ambiguous);
  }

  void Defined(const Token &MacroNameTok, const MacroDefinition &MD,
               SourceRange Range) override {
    if (!Active)
      return;
    if (const auto *MI = MD.getMacroInfo())
      recordMacroRef(MacroNameTok, *MI, RefType::Ambiguous);
  }

private:
  void recordMacroRef(const Token &Tok, const MacroInfo &MI,
                      RefType RT = RefType::Explicit) {
    if (MI.isBuiltinMacro())
      return; // __FILE__ is not a reference.
    Recorded.MacroReferences.push_back(SymbolReference{
        Tok.getLocation(),
        Macro{Tok.getIdentifierInfo(), MI.getDefinitionLoc()}, RT});
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
      : SM(CI.getSourceManager()),
        HeaderInfo(CI.getPreprocessor().getHeaderSearchInfo()), Out(Out),
        UniqueStrings(Arena) {}

  void FileChanged(SourceLocation Loc, FileChangeReason Reason,
                   SrcMgr::CharacteristicKind FileType,
                   FileID PrevFID) override {
    InMainFile = SM.isWrittenInMainFile(Loc);

    if (Reason == PPCallbacks::ExitFile) {
      // At file exit time HeaderSearchInfo is valid and can be used to
      // determine whether the file was a self-contained header or not.
      if (const FileEntry *FE = SM.getFileEntryForID(PrevFID)) {
        if (tooling::isSelfContainedHeader(FE, SM, HeaderInfo))
          Out->NonSelfContainedFiles.erase(FE->getUniqueID());
        else
          Out->NonSelfContainedFiles.insert(FE->getUniqueID());
      }
    }
  }

  void EndOfMainFile() override {
    for (auto &It : Out->IWYUExportBy) {
      llvm::sort(It.getSecond());
      It.getSecond().erase(
          std::unique(It.getSecond().begin(), It.getSecond().end()),
          It.getSecond().end());
    }
    Out->Arena = std::move(Arena);
  }

  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange /*FilenameRange*/,
                          Optional<FileEntryRef> File,
                          llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const clang::Module * /*Imported*/,
                          SrcMgr::CharacteristicKind FileKind) override {
    FileID HashFID = SM.getFileID(HashLoc);
    int HashLine = SM.getLineNumber(HashFID, SM.getFileOffset(HashLoc));
    checkForExport(HashFID, HashLine, File ? &File->getFileEntry() : nullptr);

    if (InMainFile && LastPragmaKeepInMainFileLine == HashLine)
      Out->ShouldKeep.insert(HashLine);
  }

  void checkForExport(FileID IncludingFile, int HashLine,
                      const FileEntry *IncludedHeader) {
    if (ExportStack.empty())
      return;
    auto &Top = ExportStack.back();
    if (Top.SeenAtFile != IncludingFile)
      return;
    // Make sure current include is covered by the export pragma.
    if ((Top.Block && HashLine > Top.SeenAtLine) ||
        Top.SeenAtLine == HashLine) {
      if (IncludedHeader)
        Out->IWYUExportBy[IncludedHeader->getUniqueID()].push_back(
            Top.FullPath);
      // main-file #include with export pragma should never be removed.
      if (Top.SeenAtFile == SM.getMainFileID())
        Out->ShouldKeep.insert(HashLine);
    }
    if (!Top.Block) // Pop immediately for single-line export pragma.
      ExportStack.pop_back();
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
             save(Pragma->startswith("<") || Pragma->startswith("\"")
                      ? (*Pragma)
                      : ("\"" + *Pragma + "\"").str())});
      return false;
    }
    FileID CommentFID = SM.getFileID(Range.getBegin());
    int CommentLine = SM.getLineNumber(SM.getFileID(Range.getBegin()),
                                       SM.getFileOffset(Range.getBegin()));
    // Record export pragma.
    if (Pragma->startswith("export")) {
      ExportStack.push_back(
          {CommentLine, CommentFID,
           save(SM.getFileEntryForID(CommentFID)->tryGetRealPathName()),
           false});
    } else if (Pragma->startswith("begin_exports")) {
      ExportStack.push_back(
          {CommentLine, CommentFID,
           save(SM.getFileEntryForID(CommentFID)->tryGetRealPathName()), true});
    } else if (Pragma->startswith("end_exports")) {
      // FIXME: be robust on unmatching cases. We should only pop the stack if
      // the begin_exports and end_exports is in the same file.
      if (!ExportStack.empty()) {
        assert(ExportStack.back().Block);
        ExportStack.pop_back();
      }
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
      // This code stores the last location of "IWYU pragma: keep" comment in
      // the main file, so that when next InclusionDirective is called, it will
      // know that the next inclusion is behind the IWYU pragma.
      LastPragmaKeepInMainFileLine = CommentLine;
    }
    return false;
  }

private:
  StringRef save(llvm::StringRef S) { return UniqueStrings.save(S); }

  bool InMainFile = false;
  const SourceManager &SM;
  HeaderSearch &HeaderInfo;
  PragmaIncludes *Out;
  llvm::BumpPtrAllocator Arena;
  /// Intern table for strings. Contents are on the arena.
  llvm::StringSaver UniqueStrings;
  // Track the last line "IWYU pragma: keep" was seen in the main file, 1-based.
  int LastPragmaKeepInMainFileLine = -1;
  struct ExportPragma {
    // The line number where we saw the begin_exports or export pragma.
    int SeenAtLine = 0; // 1-based line number.
    // The file where we saw the pragma.
    FileID SeenAtFile;
    // FullPath of the file SeenAtFile.
    StringRef FullPath;
    // true if it is a block begin/end_exports pragma; false if it is a
    // single-line export pragma.
    bool Block = false;
  };
  // A stack for tracking all open begin_exports or single-line export.
  std::vector<ExportPragma> ExportStack;
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

llvm::SmallVector<const FileEntry *>
PragmaIncludes::getExporters(const FileEntry *File, FileManager &FM) const {
  auto It = IWYUExportBy.find(File->getUniqueID());
  if (It == IWYUExportBy.end())
    return {};

  llvm::SmallVector<const FileEntry *> Results;
  for (auto Export : It->getSecond()) {
    // FIMXE: log the failing cases?
    if (auto FE = expectedToOptional(FM.getFileRef(Export)))
      Results.push_back(*FE);
  }
  return Results;
}

bool PragmaIncludes::isSelfContained(const FileEntry *FE) const {
  return !NonSelfContainedFiles.contains(FE->getUniqueID());
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
  ByLine[I.Line] = Index;
}

const Include *
RecordedPP::RecordedIncludes::atLine(unsigned OneBasedIndex) const {
  auto It = ByLine.find(OneBasedIndex);
  return (It == ByLine.end()) ? nullptr : &All[It->second];
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
  case Header::Verbatim:
    for (unsigned I : BySpelling.lookup(H.verbatim().trim("\"<>")))
      Result.push_back(&All[I]);
    break;
  }
  return Result;
}

std::unique_ptr<PPCallbacks> RecordedPP::record(const Preprocessor &PP) {
  return std::make_unique<PPRecorder>(*this, PP);
}

} // namespace clang::include_cleaner
