//===-- FindAllSymbolsMain.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbols.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/Path.h"

#include <map>
#include <string>
#include <vector>

using namespace clang::tooling;
using namespace llvm;
using SymbolInfo = clang::find_all_symbols::SymbolInfo;

// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static cl::OptionCategory FindAllSymbolsCategory("find_all_symbols options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

static cl::opt<std::string> OutputDir("output-dir", cl::desc(R"(
The output directory for saving the results.)"),
                                      cl::init("."),
                                      cl::cat(FindAllSymbolsCategory));

static cl::opt<std::string> MergeDir("merge-dir", cl::desc(R"(
The directory for merging symbols.)"),
                                     cl::init(""),
                                     cl::cat(FindAllSymbolsCategory));

namespace clang {
namespace find_all_symbols {

class YamlReporter
    : public clang::find_all_symbols::FindAllSymbols::ResultReporter {
public:
  ~YamlReporter() {}

  void reportResult(StringRef FileName, const SymbolInfo &Symbol) override {
    Symbols[FileName].insert(Symbol);
  }

  void Write(const std::string &Dir) {
    for (const auto &Symbol : Symbols) {
      SmallString<256> FilePath(Dir);
      llvm::sys::path::append(
          FilePath, llvm::sys::path::filename(Symbol.first) + ".yaml");
      WriteSymboInfosToFile(FilePath, Symbol.second);
    }
  }

private:
  std::map<std::string, std::set<SymbolInfo>> Symbols;
};

bool Merge(llvm::StringRef MergeDir, llvm::StringRef OutputFile) {
  std::error_code EC;
  std::set<SymbolInfo> UniqueSymbols;
  // Load all symbol files in MergeDir.
  for (llvm::sys::fs::directory_iterator Dir(MergeDir, EC), DirEnd;
       Dir != DirEnd && !EC; Dir.increment(EC)) {
    int ReadFD = 0;
    if (llvm::sys::fs::openFileForRead(Dir->path(), ReadFD)) {
      llvm::errs() << "Cann't open " << Dir->path() << "\n";
      continue;
    }
    auto Buffer = llvm::MemoryBuffer::getOpenFile(ReadFD, Dir->path(), -1);
    if (!Buffer)
      continue;
    std::vector<SymbolInfo> Symbols =
        ReadSymbolInfosFromYAML(Buffer.get()->getBuffer());
    for (const auto &Symbol : Symbols)
      UniqueSymbols.insert(Symbol);
  }

  WriteSymboInfosToFile(OutputFile, UniqueSymbols);
  return true;
}

} // namespace clang
} // namespace find_all_symbols

int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, FindAllSymbolsCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  std::vector<std::string> sources = OptionsParser.getSourcePathList();
  if (sources.empty()) {
    llvm::errs() << "Must specify at least one one source file.\n";
    return 1;
  }
  if (!MergeDir.empty()) {
    clang::find_all_symbols::Merge(MergeDir, sources[0]);
    return 0;
  }

  clang::find_all_symbols::YamlReporter Reporter;
  clang::find_all_symbols::FindAllSymbols Matcher(&Reporter);
  clang::ast_matchers::MatchFinder MatchFinder;
  Matcher.registerMatchers(&MatchFinder);
  Tool.run(newFrontendActionFactory(&MatchFinder).get());
  Reporter.Write(OutputDir);
  return 0;
}
