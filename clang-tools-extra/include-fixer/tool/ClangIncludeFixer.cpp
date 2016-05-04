//===-- ClangIncludeFixer.cpp - Standalone include fixer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryXrefsDB.h"
#include "IncludeFixer.h"
#include "XrefsDBManager.h"
#include "YamlXrefsDB.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"

using namespace clang;
using namespace llvm;

namespace {
cl::OptionCategory IncludeFixerCategory("Tool options");

enum DatabaseFormatTy {
  fixed, ///< Hard-coded mapping.
  yaml,  ///< Yaml database created by find-all-symbols.
};

cl::opt<DatabaseFormatTy> DatabaseFormat(
    "db", cl::desc("Specify input format"),
    cl::values(clEnumVal(fixed, "Hard-coded mapping"),
               clEnumVal(yaml, "Yaml database created by find-all-symbols"),
               clEnumValEnd),
    cl::init(fixed), cl::cat(IncludeFixerCategory));

cl::opt<std::string> Input("input",
                           cl::desc("String to initialize the database"),
                           cl::cat(IncludeFixerCategory));

cl::opt<bool>
    MinimizeIncludePaths("minimize-paths",
                         cl::desc("Whether to minimize added include paths"),
                         cl::init(true), cl::cat(IncludeFixerCategory));

int includeFixerMain(int argc, const char **argv) {
  tooling::CommonOptionsParser options(argc, argv, IncludeFixerCategory);
  tooling::ClangTool tool(options.getCompilations(),
                          options.getSourcePathList());

  // Set up data source.
  auto XrefsDBMgr = llvm::make_unique<include_fixer::XrefsDBManager>();
  switch (DatabaseFormat) {
  case fixed: {
    // Parse input and fill the database with it.
    // <symbol>=<header><, header...>
    // Multiple symbols can be given, separated by semicolons.
    std::map<std::string, std::vector<std::string>> XrefsMap;
    SmallVector<StringRef, 4> SemicolonSplits;
    StringRef(Input).split(SemicolonSplits, ";");
    for (StringRef Pair : SemicolonSplits) {
      auto Split = Pair.split('=');
      std::vector<std::string> Headers;
      SmallVector<StringRef, 4> CommaSplits;
      Split.second.split(CommaSplits, ",");
      for (StringRef Header : CommaSplits)
        Headers.push_back(Header.trim());
      XrefsMap[Split.first.trim()] = std::move(Headers);
    }
    XrefsDBMgr->addXrefsDB(
        llvm::make_unique<include_fixer::InMemoryXrefsDB>(std::move(XrefsMap)));
    break;
  }
  case yaml: {
    XrefsDBMgr->addXrefsDB(
        llvm::make_unique<include_fixer::YamlXrefsDB>(Input));
    break;
  }
  }

  // Now run our tool.
  std::vector<tooling::Replacement> Replacements;
  include_fixer::IncludeFixerActionFactory Factory(*XrefsDBMgr, Replacements,
                                                   MinimizeIncludePaths);

  tool.run(&Factory); // Always succeeds.

  // Set up a new source manager for applying the resulting replacements.
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions);
  DiagnosticsEngine Diagnostics(new DiagnosticIDs, &*DiagOpts);
  TextDiagnosticPrinter DiagnosticPrinter(outs(), &*DiagOpts);
  SourceManager SM(Diagnostics, tool.getFiles());
  Diagnostics.setClient(&DiagnosticPrinter, false);

  // Write replacements to disk.
  Rewriter Rewrites(SM, LangOptions());
  tooling::applyAllReplacements(Replacements, Rewrites);
  return Rewrites.overwriteChangedFiles();
}

} // namespace

int main(int argc, const char **argv) {
  return includeFixerMain(argc, argv);
}
