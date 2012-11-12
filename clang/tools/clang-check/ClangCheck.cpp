//===--- tools/clang-check/ClangCheck.cpp - Clang check tool --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements a clang-check tool that runs clang based on the info
//  stored in a compilation database.
//
//  This tool uses the Clang Tooling infrastructure, see
//    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
//  for details on setting it up with LLVM source tree.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Rewrite/Frontend/FixItRewriter.h"
#include "clang/Rewrite/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm;

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
    "\tFor example, to run clang-check on all files in a subtree of the\n"
    "\tsource tree, use:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check\n"
    "\n"
    "\tor using a specific build path:\n"
    "\n"
    "\t  find path/in/subtree -name '*.cpp'|xargs clang-check -p build/path\n"
    "\n"
    "\tNote, that path/in/subtree and current directory should follow the\n"
    "\trules described above.\n"
    "\n"
);

static OwningPtr<OptTable> Options(createDriverOptTable());
static cl::opt<bool> ASTDump(
    "ast-dump",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_dump)));
static cl::opt<bool> ASTList(
    "ast-list",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_list)));
static cl::opt<bool> ASTPrint(
    "ast-print",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_print)));
static cl::opt<std::string> ASTDumpFilter(
    "ast-dump-filter",
    cl::desc(Options->getOptionHelpText(options::OPT_ast_dump_filter)));

static cl::opt<bool> Fixit(
    "fixit",
    cl::desc(Options->getOptionHelpText(options::OPT_fixit)));
static cl::opt<bool> FixWhatYouCan(
    "fix-what-you-can",
    cl::desc(Options->getOptionHelpText(options::OPT_fix_what_you_can)));

namespace {

// FIXME: Move FixItRewriteInPlace from lib/Rewrite/Frontend/FrontendActions.cpp
// into a header file and reuse that.
class FixItOptions : public clang::FixItOptions {
public:
  FixItOptions() {
    FixWhatYouCan = ::FixWhatYouCan;
  }

  std::string RewriteFilename(const std::string& filename, int &fd) {
    assert(llvm::sys::path::is_absolute(filename) &&
           "clang-fixit expects absolute paths only.");

    // We don't need to do permission checking here since clang will diagnose
    // any I/O errors itself.

    fd = -1;  // No file descriptor for file.

    return filename;
  }
};

/// \brief Subclasses \c clang::FixItRewriter to not count fixed errors/warnings
/// in the final error counts.
///
/// This has the side-effect that clang-check -fixit exits with code 0 on
/// successfully fixing all errors.
class FixItRewriter : public clang::FixItRewriter {
public:
  FixItRewriter(clang::DiagnosticsEngine& Diags,
                clang::SourceManager& SourceMgr,
                const clang::LangOptions& LangOpts,
                clang::FixItOptions* FixItOpts)
      : clang::FixItRewriter(Diags, SourceMgr, LangOpts, FixItOpts) {
  }

  virtual bool IncludeInDiagnosticCounts() const { return false; }
};

/// \brief Subclasses \c clang::FixItAction so that we can install the custom
/// \c FixItRewriter.
class FixItAction : public clang::FixItAction {
public:
  virtual bool BeginSourceFileAction(clang::CompilerInstance& CI,
                                     StringRef Filename) {
    FixItOpts.reset(new FixItOptions);
    Rewriter.reset(new FixItRewriter(CI.getDiagnostics(), CI.getSourceManager(),
                                     CI.getLangOpts(), FixItOpts.get()));
    return true;
  }
};

} // namespace

// Anonymous namespace here causes problems with gcc <= 4.4 on MacOS 10.6.
// "Non-global symbol: ... can't be a weak_definition"
namespace clang_check {
class ClangCheckActionFactory {
public:
  clang::ASTConsumer *newASTConsumer() {
    if (ASTList)
      return clang::CreateASTDeclNodeLister();
    if (ASTDump)
      return clang::CreateASTDumper(ASTDumpFilter);
    if (ASTPrint)
      return clang::CreateASTPrinter(&llvm::outs(), ASTDumpFilter);
    return new clang::ASTConsumer();
  }
};
}

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv);
  ClangTool Tool(OptionsParser.GetCompilations(),
                 OptionsParser.GetSourcePathList());
  if (Fixit)
    return Tool.run(newFrontendActionFactory<FixItAction>());
  clang_check::ClangCheckActionFactory Factory;
  return Tool.run(newFrontendActionFactory(&Factory));
}
