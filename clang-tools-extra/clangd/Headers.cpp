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
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Tooling/CompilationDatabase.h"
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
llvm::Expected<std::string>
calculateIncludePath(llvm::StringRef File, llvm::StringRef Code,
                     const HeaderFile &DeclaringHeader,
                     const HeaderFile &InsertedHeader,
                     const tooling::CompileCommand &CompileCommand,
                     IntrusiveRefCntPtr<vfs::FileSystem> FS) {
  assert(llvm::sys::path::is_absolute(File));
  assert(DeclaringHeader.valid() && InsertedHeader.valid());
  if (File == DeclaringHeader.File || File == InsertedHeader.File)
    return "";
  FS->setCurrentWorkingDirectory(CompileCommand.Directory);

  // Set up a CompilerInstance and create a preprocessor to collect existing
  // #include headers in \p Code. Preprocesor also provides HeaderSearch with
  // which we can calculate the shortest include path for \p Header.
  std::vector<const char *> Argv;
  for (const auto &S : CompileCommand.CommandLine)
    Argv.push_back(S.c_str());
  IgnoringDiagConsumer IgnoreDiags;
  auto CI = clang::createInvocationFromCommandLine(
      Argv,
      CompilerInstance::createDiagnostics(new DiagnosticOptions(), &IgnoreDiags,
                                          false),
      FS);
  if (!CI)
    return llvm::make_error<llvm::StringError>(
        "Failed to create a compiler instance for " + File,
        llvm::inconvertibleErrorCode());
  CI->getFrontendOpts().DisableFree = false;
  // Parse the main file to get all existing #includes in the file, and then we
  // can make sure the same header (even with different include path) is not
  // added more than once.
  CI->getPreprocessorOpts().SingleFileParseMode = true;

  // The diagnostic options must be set before creating a CompilerInstance.
  CI->getDiagnosticOpts().IgnoreWarnings = true;
  auto Clang = prepareCompilerInstance(
      std::move(CI), /*Preamble=*/nullptr,
      llvm::MemoryBuffer::getMemBuffer(Code, File),
      std::make_shared<PCHContainerOperations>(), FS, IgnoreDiags);

  if (Clang->getFrontendOpts().Inputs.empty())
    return llvm::make_error<llvm::StringError>(
        "Empty frontend action inputs empty for file " + File,
        llvm::inconvertibleErrorCode());
  PreprocessOnlyAction Action;
  if (!Action.BeginSourceFile(*Clang, Clang->getFrontendOpts().Inputs[0]))
    return llvm::make_error<llvm::StringError>(
        "Failed to begin preprocessor only action for file " + File,
        llvm::inconvertibleErrorCode());
  std::vector<Inclusion> Inclusions;
  Clang->getPreprocessor().addPPCallbacks(collectInclusionsInMainFileCallback(
      Clang->getSourceManager(),
      [&Inclusions](Inclusion Inc) { Inclusions.push_back(std::move(Inc)); }));
  if (!Action.Execute())
    return llvm::make_error<llvm::StringError>(
        "Failed to execute preprocessor only action for file " + File,
        llvm::inconvertibleErrorCode());
  llvm::StringSet<> IncludedHeaders;
  for (const auto &Inc : Inclusions) {
    IncludedHeaders.insert(Inc.Written);
    if (!Inc.Resolved.empty())
      IncludedHeaders.insert(Inc.Resolved);
  }
  auto Included = [&](llvm::StringRef Header) {
    return IncludedHeaders.find(Header) != IncludedHeaders.end();
  };
  if (Included(DeclaringHeader.File) || Included(InsertedHeader.File))
    return "";

  auto &HeaderSearchInfo = Clang->getPreprocessor().getHeaderSearchInfo();
  bool IsSystem = false;

  if (InsertedHeader.Verbatim)
    return InsertedHeader.File;

  std::string Suggested = HeaderSearchInfo.suggestPathToFileForDiagnostics(
      InsertedHeader.File, CompileCommand.Directory, &IsSystem);
  if (IsSystem)
    Suggested = "<" + Suggested + ">";
  else
    Suggested = "\"" + Suggested + "\"";

  log("Suggested #include for " + InsertedHeader.File + " is: " + Suggested);
  return Suggested;
}

} // namespace clangd
} // namespace clang
