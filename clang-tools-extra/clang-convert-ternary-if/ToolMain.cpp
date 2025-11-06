//===--- ToolMain.cpp - Entry point for clang-convert-ternary-if ----------===//
//
// This tool runs the refactoring logic defined in ConvertTernaryIf.cpp.
//
// Usage:
//   clang-convert-ternary-if <source-file> --
//
// It prints the rewritten (refactored) source code to stdout.
//
//===----------------------------------------------------------------------===//

#include "ConvertTernaryIf.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::convertternary;
using namespace llvm;

static llvm::cl::OptionCategory ToolCategory("convert-ternary-if options");

int main(int argc, const char **argv) {
  // Parse command-line options
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, ToolCategory, cl::ZeroOrMore);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  // Set up the Rewriter and the Matcher
  clang::Rewriter Rewrite;
  ast_matchers::MatchFinder Finder;
  ConvertTernaryIfCallback Callback(Rewrite);
  setupMatchers(Finder, Callback);

  llvm::outs() << "=== Running clang-convert-ternary-if ===\n";
  int Result = Tool.run(newFrontendActionFactory(&Finder).get());

  if (Result != 0) {
    llvm::errs() << "Error: Tool execution failed.\n";
    return Result;
  }

  // No changes?
  if (Rewrite.buffer_begin() == Rewrite.buffer_end()) {
    llvm::outs() << "No changes made.\n";
    return 0;
  }

  llvm::outs() << "\n=== Rewritten Files ===\n";

  // Print all rewritten files
  for (auto It = Rewrite.buffer_begin(); It != Rewrite.buffer_end(); ++It) {
    clang::FileID FID = It->first;
    const llvm::RewriteBuffer &RewriteBuf = It->second;
    const clang::SourceManager &SM = Rewrite.getSourceMgr();

    // Get the filename safely
    llvm::StringRef FileName = SM.getFilename(SM.getLocForStartOfFile(FID));
    if (FileName.empty())
      FileName = "<unknown file>";

    llvm::outs() << "\n--- " << FileName << " ---\n";
    RewriteBuf.write(llvm::outs());
    llvm::outs() << "\n";
  }

  llvm::outs() << "\n=== Refactoring complete ===\n";
  return 0;
}

