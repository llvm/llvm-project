//===-- FindAllSymbolsMain.cpp - find all symbols tool ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FindAllSymbolsAction.h"
#include "STLPostfixHeaderMap.h"
#include "SymbolInfo.h"
#include "SymbolReporter.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
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

class YamlReporter : public SymbolReporter {
public:
  void reportSymbols(StringRef FileName,
                     const SymbolInfo::SignalMap &Symbols) override {
    int FD;
    SmallString<128> ResultPath;
    llvm::sys::fs::createUniqueFile(
        OutputDir + "/" + llvm::sys::path::filename(FileName) + "-%%%%%%.yaml",
        FD, ResultPath);
    llvm::raw_fd_ostream OS(FD, /*shouldClose=*/true);
    WriteSymbolInfosToStream(OS, Symbols);
  }
};

bool Merge(llvm::StringRef MergeDir, llvm::StringRef OutputFile) {
  std::error_code EC;
  SymbolInfo::SignalMap Symbols;
  std::mutex SymbolMutex;
  auto AddSymbols = [&](ArrayRef<SymbolAndSignals> NewSymbols) {
    // Synchronize set accesses.
    std::unique_lock<std::mutex> LockGuard(SymbolMutex);
    for (const auto &Symbol : NewSymbols) {
      Symbols[Symbol.Symbol] += Symbol.Signals;
    }
  };

  // Load all symbol files in MergeDir.
  {
    llvm::DefaultThreadPool Pool;
    for (llvm::sys::fs::directory_iterator Dir(MergeDir, EC), DirEnd;
         Dir != DirEnd && !EC; Dir.increment(EC)) {
      // Parse YAML files in parallel.
      Pool.async(
          [&AddSymbols](std::string Path) {
            auto Buffer = llvm::MemoryBuffer::getFile(Path);
            if (!Buffer) {
              llvm::errs() << "Can't open " << Path << "\n";
              return;
            }
            std::vector<SymbolAndSignals> Symbols =
                ReadSymbolInfosFromYAML(Buffer.get()->getBuffer());
            for (auto &Symbol : Symbols) {
              // Only count one occurrence per file, to avoid spam.
              Symbol.Signals.Seen = std::min(Symbol.Signals.Seen, 1u);
              Symbol.Signals.Used = std::min(Symbol.Signals.Used, 1u);
            }
            // FIXME: Merge without creating such a heavy contention point.
            AddSymbols(Symbols);
          },
          Dir->path());
    }
  }

  llvm::raw_fd_ostream OS(OutputFile, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Can't open '" << OutputFile << "': " << EC.message()
                 << '\n';
    return false;
  }
  WriteSymbolInfosToStream(OS, Symbols);
  return true;
}

} // namespace clang
} // namespace find_all_symbols

int main(int argc, const char **argv) {
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, FindAllSymbolsCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  CommonOptionsParser &OptionsParser = ExpectedParser.get();
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

  auto Factory =
      std::make_unique<clang::find_all_symbols::FindAllSymbolsActionFactory>(
          &Reporter, clang::find_all_symbols::getSTLPostfixHeaderMap());
  return Tool.run(Factory.get());
}
