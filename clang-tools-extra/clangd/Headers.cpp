//===--- Headers.cpp - Include headers ---------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Headers.h"
#include "Compiler.h"
#include "Logger.h"
#include "SourceCode.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/HeaderSearch.h"
#include "llvm/Support/Path.h"

namespace clang {
namespace clangd {
namespace {

class RecordHeaders : public PPCallbacks {
public:
  RecordHeaders(const SourceManager &SM,
                std::function<void(Inclusion)> Callback)
      : SM(SM), Callback(std::move(Callback)) {}

  // Record existing #includes - both written and resolved paths. Only #includes
  // in the main file are collected.
  void InclusionDirective(SourceLocation HashLoc, const Token & /*IncludeTok*/,
                          llvm::StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange, const FileEntry *File,
                          llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const Module * /*Imported*/,
                          SrcMgr::CharacteristicKind /*FileType*/) override {
    // Only inclusion directives in the main file make sense. The user cannot
    // select directives not in the main file.
    if (HashLoc.isInvalid() || !SM.isInMainFile(HashLoc))
      return;
    std::string Written =
        (IsAngled ? "<" + FileName + ">" : "\"" + FileName + "\"").str();
    std::string Resolved = (!File || File->tryGetRealPathName().empty())
                               ? ""
                               : File->tryGetRealPathName();
    Callback({halfOpenToRange(SM, FilenameRange), Written, Resolved});
  }

private:
  const SourceManager &SM;
  std::function<void(Inclusion)> Callback;
};

} // namespace

bool isLiteralInclude(llvm::StringRef Include) {
  return Include.startswith("<") || Include.startswith("\"");
}

bool HeaderFile::valid() const {
  return (Verbatim && isLiteralInclude(File)) ||
         (!Verbatim && llvm::sys::path::is_absolute(File));
}

std::unique_ptr<PPCallbacks>
collectInclusionsInMainFileCallback(const SourceManager &SM,
                                    std::function<void(Inclusion)> Callback) {
  return llvm::make_unique<RecordHeaders>(SM, std::move(Callback));
}

/// FIXME(ioeric): we might not want to insert an absolute include path if the
/// path is not shortened.
bool IncludeInserter::shouldInsertInclude(
    const HeaderFile &DeclaringHeader, const HeaderFile &InsertedHeader) const {
  assert(DeclaringHeader.valid() && InsertedHeader.valid());
  if (FileName == DeclaringHeader.File || FileName == InsertedHeader.File)
    return false;
  llvm::StringSet<> IncludedHeaders;
  for (const auto &Inc : Inclusions) {
    IncludedHeaders.insert(Inc.Written);
    if (!Inc.Resolved.empty())
      IncludedHeaders.insert(Inc.Resolved);
  }
  auto Included = [&](llvm::StringRef Header) {
    return IncludedHeaders.find(Header) != IncludedHeaders.end();
  };
  return !Included(DeclaringHeader.File) && !Included(InsertedHeader.File);
}

std::string
IncludeInserter::calculateIncludePath(const HeaderFile &DeclaringHeader,
                                      const HeaderFile &InsertedHeader) const {
  assert(DeclaringHeader.valid() && InsertedHeader.valid());
  if (InsertedHeader.Verbatim)
    return InsertedHeader.File;
  bool IsSystem = false;
  std::string Suggested = HeaderSearchInfo.suggestPathToFileForDiagnostics(
      InsertedHeader.File, BuildDir, &IsSystem);
  if (IsSystem)
    Suggested = "<" + Suggested + ">";
  else
    Suggested = "\"" + Suggested + "\"";
  return Suggested;
}

Optional<TextEdit> IncludeInserter::insert(StringRef VerbatimHeader) const {
  Optional<TextEdit> Edit = None;
  if (auto Insertion = Inserter.insert(VerbatimHeader.trim("\"<>"),
                                       VerbatimHeader.startswith("<")))
    Edit = replacementToEdit(Code, *Insertion);
  return Edit;
}

} // namespace clangd
} // namespace clang
