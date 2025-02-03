//===--- IncludeCleaner.cpp - standalone tool for include analysis --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "clang-include-cleaner/Analysis.h"
#include "clang-include-cleaner/Record.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace clang {
namespace include_cleaner {
namespace {
namespace cl = llvm::cl;

llvm::StringRef Overview = llvm::StringLiteral(R"(
clang-include-cleaner analyzes the #include directives in source code.

It suggests removing headers that the code is not using.
It suggests inserting headers that the code relies on, but does not include.
These changes make the file more self-contained and (at scale) make the codebase
easier to reason about and modify.

The tool operates on *working* source code. This means it can suggest including
headers that are only indirectly included, but cannot suggest those that are
missing entirely. (clang-include-fixer can do this).
)")
                               .trim();

cl::OptionCategory IncludeCleaner("clang-include-cleaner");

cl::opt<std::string> HTMLReportPath{
    "html",
    cl::desc("Specify an output filename for an HTML report. "
             "This describes both recommendations and reasons for changes."),
    cl::cat(IncludeCleaner),
};

cl::opt<std::string> OnlyHeaders{
    "only-headers",
    cl::desc("A comma-separated list of regexes to match against suffix of a "
             "header. Only headers that match will be analyzed."),
    cl::init(""),
    cl::cat(IncludeCleaner),
};

cl::opt<std::string> IgnoreHeaders{
    "ignore-headers",
    cl::desc("A comma-separated list of regexes to match against suffix of a "
             "header, and disable analysis if matched."),
    cl::init(""),
    cl::cat(IncludeCleaner),
};

enum class PrintStyle { Changes, Final };
cl::opt<PrintStyle> Print{
    "print",
    cl::values(
        clEnumValN(PrintStyle::Changes, "changes", "Print symbolic changes"),
        clEnumValN(PrintStyle::Final, "", "Print final code")),
    cl::ValueOptional,
    cl::init(PrintStyle::Final),
    cl::desc("Print the list of headers to insert and remove"),
    cl::cat(IncludeCleaner),
};

cl::opt<bool> Edit{
    "edit",
    cl::desc("Apply edits to analyzed source files"),
    cl::cat(IncludeCleaner),
};

cl::opt<bool> Insert{
    "insert",
    cl::desc("Allow header insertions"),
    cl::init(true),
    cl::cat(IncludeCleaner),
};
cl::opt<bool> Remove{
    "remove",
    cl::desc("Allow header removals"),
    cl::init(true),
    cl::cat(IncludeCleaner),
};

std::atomic<unsigned> Errors = ATOMIC_VAR_INIT(0);

format::FormatStyle getStyle(llvm::StringRef Filename) {
  auto S = format::getStyle(format::DefaultFormatStyle, Filename,
                            format::DefaultFallbackStyle);
  if (!S || !S->isCpp()) {
    consumeError(S.takeError());
    return format::getLLVMStyle();
  }
  return std::move(*S);
}

class Action : public clang::ASTFrontendAction {
public:
  Action(llvm::function_ref<bool(llvm::StringRef)> HeaderFilter,
         llvm::StringMap<std::string> &EditedFiles)
      : HeaderFilter(HeaderFilter), EditedFiles(EditedFiles) {}

private:
  RecordedAST AST;
  RecordedPP PP;
  PragmaIncludes PI;
  llvm::function_ref<bool(llvm::StringRef)> HeaderFilter;
  llvm::StringMap<std::string> &EditedFiles;

  bool BeginInvocation(CompilerInstance &CI) override {
    // We only perform include-cleaner analysis. So we disable diagnostics that
    // won't affect our analysis to make the tool more robust against
    // in-development code.
    CI.getLangOpts().ModulesDeclUse = false;
    CI.getLangOpts().ModulesStrictDeclUse = false;
    return true;
  }

  void ExecuteAction() override {
    const auto &CI = getCompilerInstance();

    // Disable all warnings when running include-cleaner, as we are only
    // interested in include-cleaner related findings. This makes the tool both
    // more resilient around in-development code, and possibly faster as we
    // skip some extra analysis.
    auto &Diags = CI.getDiagnostics();
    Diags.setEnableAllWarnings(false);
    Diags.setSeverityForAll(clang::diag::Flavor::WarningOrError,
                            clang::diag::Severity::Ignored);
    auto &P = CI.getPreprocessor();
    P.addPPCallbacks(PP.record(P));
    PI.record(getCompilerInstance());
    ASTFrontendAction::ExecuteAction();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef File) override {
    return AST.record();
  }

  void EndSourceFile() override {
    const auto &SM = getCompilerInstance().getSourceManager();
    if (SM.getDiagnostics().hasUncompilableErrorOccurred()) {
      llvm::errs()
          << "Skipping file " << getCurrentFile()
          << " due to compiler errors. clang-include-cleaner expects to "
             "work on compilable source code.\n";
      return;
    }

    if (!HTMLReportPath.empty())
      writeHTML();

    // Source File's path of compiler invocation, converted to absolute path.
    llvm::SmallString<256> AbsPath(
        SM.getFileEntryRefForID(SM.getMainFileID())->getName());
    assert(!AbsPath.empty() && "Main file path not known?");
    SM.getFileManager().makeAbsolutePath(AbsPath);
    llvm::StringRef Code = SM.getBufferData(SM.getMainFileID());

    auto Results =
        analyze(AST.Roots, PP.MacroReferences, PP.Includes, &PI,
                getCompilerInstance().getPreprocessor(), HeaderFilter);
    if (!Insert)
      Results.Missing.clear();
    if (!Remove)
      Results.Unused.clear();
    std::string Final = fixIncludes(Results, AbsPath, Code, getStyle(AbsPath));

    if (Print.getNumOccurrences()) {
      switch (Print) {
      case PrintStyle::Changes:
        for (const Include *I : Results.Unused)
          llvm::outs() << "- " << I->quote() << " @Line:" << I->Line << "\n";
        for (const auto &[I, _] : Results.Missing)
          llvm::outs() << "+ " << I << "\n";
        break;
      case PrintStyle::Final:
        llvm::outs() << Final;
        break;
      }
    }

    if (!Results.Missing.empty() || !Results.Unused.empty())
      EditedFiles.try_emplace(AbsPath, Final);
  }

  void writeHTML() {
    std::error_code EC;
    llvm::raw_fd_ostream OS(HTMLReportPath, EC);
    if (EC) {
      llvm::errs() << "Unable to write HTML report to " << HTMLReportPath
                   << ": " << EC.message() << "\n";
      ++Errors;
      return;
    }
    writeHTMLReport(AST.Ctx->getSourceManager().getMainFileID(), PP.Includes,
                    AST.Roots, PP.MacroReferences, *AST.Ctx,
                    getCompilerInstance().getPreprocessor(), &PI, OS);
  }
};
class ActionFactory : public tooling::FrontendActionFactory {
public:
  ActionFactory(llvm::function_ref<bool(llvm::StringRef)> HeaderFilter)
      : HeaderFilter(HeaderFilter) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    return std::make_unique<Action>(HeaderFilter, EditedFiles);
  }

  const llvm::StringMap<std::string> &editedFiles() const {
    return EditedFiles;
  }

private:
  llvm::function_ref<bool(llvm::StringRef)> HeaderFilter;
  // Map from file name to final code with the include edits applied.
  llvm::StringMap<std::string> EditedFiles;
};

// Compiles a regex list into a function that return true if any match a header.
// Prints and returns nullptr if any regexes are invalid.
std::function<bool(llvm::StringRef)> matchesAny(llvm::StringRef RegexFlag) {
  auto FilterRegs = std::make_shared<std::vector<llvm::Regex>>();
  llvm::SmallVector<llvm::StringRef> Headers;
  RegexFlag.split(Headers, ',', -1, /*KeepEmpty=*/false);
  for (auto HeaderPattern : Headers) {
    std::string AnchoredPattern = "(" + HeaderPattern.str() + ")$";
    llvm::Regex CompiledRegex(AnchoredPattern);
    std::string RegexError;
    if (!CompiledRegex.isValid(RegexError)) {
      llvm::errs() << llvm::formatv("Invalid regular expression '{0}': {1}\n",
                                    HeaderPattern, RegexError);
      return nullptr;
    }
    FilterRegs->push_back(std::move(CompiledRegex));
  }
  return [FilterRegs](llvm::StringRef Path) {
    for (const auto &F : *FilterRegs) {
      if (F.match(Path))
        return true;
    }
    return false;
  };
}

std::function<bool(llvm::StringRef)> headerFilter() {
  auto OnlyMatches = matchesAny(OnlyHeaders);
  auto IgnoreMatches = matchesAny(IgnoreHeaders);
  if (!OnlyMatches || !IgnoreMatches)
    return nullptr;

  return [OnlyMatches, IgnoreMatches](llvm::StringRef Header) {
    if (!OnlyHeaders.empty() && !OnlyMatches(Header))
      return true;
    if (!IgnoreHeaders.empty() && IgnoreMatches(Header))
      return true;
    return false;
  };
}

// Maps absolute path of each files of each compilation commands to the
// absolute path of the input file.
llvm::Expected<std::map<std::string, std::string>>
mapInputsToAbsPaths(clang::tooling::CompilationDatabase &CDB,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> VFS,
                    const std::vector<std::string> &Inputs) {
  std::map<std::string, std::string> CDBToAbsPaths;
  // Factory.editedFiles()` will contain the final code, along with the
  // path given in the compilation database. That path can be
  // absolute or relative, and if it is relative, it is relative to the
  // "Directory" field in the compilation database. We need to make it
  // absolute to write the final code to the correct path.
  for (auto &Source : Inputs) {
    llvm::SmallString<256> AbsPath(Source);
    if (auto Err = VFS->makeAbsolute(AbsPath)) {
      llvm::errs() << "Failed to get absolute path for " << Source << " : "
                   << Err.message() << '\n';
      return llvm::errorCodeToError(Err);
    }
    std::vector<clang::tooling::CompileCommand> Cmds =
        CDB.getCompileCommands(AbsPath);
    if (Cmds.empty()) {
      // It should be found in the compilation database, even user didn't
      // specify the compilation database, the `FixedCompilationDatabase` will
      // create an entry from the arguments. So it is an error if we can't
      // find the compile commands.
      std::string ErrorMsg =
          llvm::formatv("No compile commands found for {0}", AbsPath).str();
      llvm::errs() << ErrorMsg << '\n';
      return llvm::make_error<llvm::StringError>(
          ErrorMsg, llvm::inconvertibleErrorCode());
    }
    for (const auto &Cmd : Cmds) {
      llvm::SmallString<256> CDBPath(Cmd.Filename);
      std::string Directory(Cmd.Directory);
      llvm::sys::fs::make_absolute(Cmd.Directory, CDBPath);
      CDBToAbsPaths[std::string(CDBPath)] = std::string(AbsPath);
    }
  }
  return CDBToAbsPaths;
}

} // namespace
} // namespace include_cleaner
} // namespace clang

int main(int argc, const char **argv) {
  using namespace clang::include_cleaner;

  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
  auto OptionsParser =
      clang::tooling::CommonOptionsParser::create(argc, argv, IncludeCleaner);
  if (!OptionsParser) {
    llvm::errs() << toString(OptionsParser.takeError());
    return 1;
  }

  if (OptionsParser->getSourcePathList().size() != 1) {
    std::vector<cl::Option *> IncompatibleFlags = {&HTMLReportPath, &Print};
    for (const auto *Flag : IncompatibleFlags) {
      if (Flag->getNumOccurrences()) {
        llvm::errs() << "-" << Flag->ArgStr << " requires a single input file";
        return 1;
      }
    }
  }

  auto VFS = llvm::vfs::getRealFileSystem();
  auto &CDB = OptionsParser->getCompilations();
  // CDBToAbsPaths is a map from the path in the compilation database to the
  // writable absolute path of the file.
  auto CDBToAbsPaths =
      mapInputsToAbsPaths(CDB, VFS, OptionsParser->getSourcePathList());
  if (!CDBToAbsPaths)
    return 1;

  clang::tooling::ClangTool Tool(CDB, OptionsParser->getSourcePathList());

  auto HeaderFilter = headerFilter();
  if (!HeaderFilter)
    return 1; // error already reported.
  ActionFactory Factory(HeaderFilter);
  auto ErrorCode = Tool.run(&Factory);
  if (Edit) {
    for (const auto &NameAndContent : Factory.editedFiles()) {
      llvm::StringRef FileName = NameAndContent.first();
      if (auto It = CDBToAbsPaths->find(FileName.str());
          It != CDBToAbsPaths->end())
        FileName = It->second;

      const std::string &FinalCode = NameAndContent.second;
      if (auto Err = llvm::writeToOutput(
              FileName, [&](llvm::raw_ostream &OS) -> llvm::Error {
                OS << FinalCode;
                return llvm::Error::success();
              })) {
        llvm::errs() << "Failed to apply edits to " << FileName << ": "
                     << toString(std::move(Err)) << "\n";
        ++Errors;
      }
    }
  }
  return ErrorCode || Errors != 0;
}
