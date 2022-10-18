//===--- Record.cpp - Record compiler events ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Record.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/PPCallbacks.h"
#include "clang/Lex/Preprocessor.h"

namespace clang::include_cleaner {

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

} // namespace clang::include_cleaner
