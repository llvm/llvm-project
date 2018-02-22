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
  RecordHeaders(std::set<std::string> &Headers) : Headers(Headers) {}

  void InclusionDirective(SourceLocation /*HashLoc*/,
                          const Token & /*IncludeTok*/,
                          llvm::StringRef /*FileName*/, bool /*IsAngled*/,
                          CharSourceRange /*FilenameRange*/,
                          const FileEntry *File, llvm::StringRef /*SearchPath*/,
                          llvm::StringRef /*RelativePath*/,
                          const Module * /*Imported*/) override {
    if (File != nullptr && !File->tryGetRealPathName().empty())
      Headers.insert(File->tryGetRealPathName());
  }

private:
  std::set<std::string> &Headers;
};

} // namespace

/// FIXME(ioeric): we might not want to insert an absolute include path if the
/// path is not shortened.
llvm::Expected<std::string>
calculateIncludePath(llvm::StringRef File, llvm::StringRef Code,
                     llvm::StringRef Header,
                     const tooling::CompileCommand &CompileCommand,
                     IntrusiveRefCntPtr<vfs::FileSystem> FS) {
  assert(llvm::sys::path::is_absolute(File) &&
         llvm::sys::path::is_absolute(Header));

  if (File == Header)
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
  std::set<std::string> ExistingHeaders;
  Clang->getPreprocessor().addPPCallbacks(
      llvm::make_unique<RecordHeaders>(ExistingHeaders));
  if (!Action.Execute())
    return llvm::make_error<llvm::StringError>(
        "Failed to execute preprocessor only action for file " + File,
        llvm::inconvertibleErrorCode());
  if (ExistingHeaders.find(Header) != ExistingHeaders.end()) {
    return llvm::make_error<llvm::StringError>(
        Header + " is already included in " + File,
        llvm::inconvertibleErrorCode());
  }

  auto &HeaderSearchInfo = Clang->getPreprocessor().getHeaderSearchInfo();
  bool IsSystem = false;
  std::string Suggested = HeaderSearchInfo.suggestPathToFileForDiagnostics(
      Header, CompileCommand.Directory, &IsSystem);
  if (IsSystem)
    Suggested = "<" + Suggested + ">";
  else
    Suggested = "\"" + Suggested + "\"";

  log("Suggested #include for " + Header + " is: " + Suggested);
  return Suggested;
}

} // namespace clangd
} // namespace clang
