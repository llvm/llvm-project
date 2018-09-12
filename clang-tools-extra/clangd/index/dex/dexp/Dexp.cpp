//===--- Dexp.cpp - Dex EXPloration tool ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple interactive tool which can be used to manually
// evaluate symbol search quality of Clangd index.
//
//===----------------------------------------------------------------------===//

#include "../../../index/SymbolYAML.h"
#include "../Dex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

using clang::clangd::FuzzyFindRequest;
using clang::clangd::loadIndex;
using clang::clangd::Symbol;
using clang::clangd::SymbolIndex;
using llvm::StringRef;

namespace {

llvm::cl::opt<std::string>
    SymbolCollection("symbol-collection-file",
                     llvm::cl::desc("Path to the file with symbol collection"),
                     llvm::cl::Positional, llvm::cl::Required);

static const std::string Overview = R"(
This is an **experimental** interactive tool to process user-provided search
queries over given symbol collection obtained via global-symbol-builder. The
tool can be used to evaluate search quality of existing index implementations
and manually construct non-trivial test cases.

Type use "help" request to get information about the details.
)";

void reportTime(StringRef Name, llvm::function_ref<void()> F) {
  const auto TimerStart = std::chrono::high_resolution_clock::now();
  F();
  const auto TimerStop = std::chrono::high_resolution_clock::now();
  const auto Duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      TimerStop - TimerStart);
  llvm::outs() << llvm::formatv("{0} took {1:ms+n}.\n", Name, Duration);
}

void fuzzyFind(llvm::StringRef UnqualifiedName, const SymbolIndex &Index) {
  FuzzyFindRequest Request;
  Request.MaxCandidateCount = 10;
  Request.Query = UnqualifiedName;
  // FIXME(kbobyrev): Print symbol final scores to see the distribution.
  static const auto OutputFormat = "{0,-4} | {1,-40} | {2,-25}\n";
  llvm::outs() << llvm::formatv(OutputFormat, "Rank", "Symbol ID",
                                "Symbol Name");
  size_t Rank = 0;
  Index.fuzzyFind(Request, [&](const Symbol &Sym) {
    llvm::outs() << llvm::formatv(OutputFormat, Rank++, Sym.ID.str(), Sym.Name);
  });
}

static const std::string HelpMessage = R"(dexp commands:

> find Name

Constructs fuzzy find request given unqualified symbol name and returns top 10
symbols retrieved from index.

> lookup SymbolID

Retrieves symbol names given USR.
)";

void help() { llvm::outs() << HelpMessage; }

void lookup(StringRef USR, const SymbolIndex &Index) {
  llvm::DenseSet<clang::clangd::SymbolID> IDs{clang::clangd::SymbolID{USR}};
  clang::clangd::LookupRequest Request{IDs};
  bool FoundSymbol = false;
  Index.lookup(Request, [&](const Symbol &Sym) {
    if (!FoundSymbol)
      FoundSymbol = true;
    llvm::outs() << SymbolToYAML(Sym);
  });
  if (!FoundSymbol)
    llvm::outs() << "not found\n";
}

// FIXME(kbobyrev): Make this an actual REPL: probably use LLVM Command Line
// library for parsing flags and arguments.
// FIXME(kbobyrev): Ideas for commands:
// * symbol lookup: print out symbol in YAML format given SymbolID
// * find symbol references: print set of reference locations
// * load/swap/reload index: this would make it possible to get rid of llvm::cl
//   usages in the tool driver and actually use llvm::cl library in the REPL.
// * show posting list density histogram (our dump data somewhere so that user
//   could build one)
// * show number of tokens of each kind
// * print out tokens with the most dense posting lists
// * print out tokens with least dense posting lists
void dispatch(StringRef Request, const SymbolIndex &Index) {
  llvm::SmallVector<StringRef, 2> Arguments;
  Request.split(Arguments, ' ');
  if (Arguments.empty()) {
    llvm::outs() << "Request can not be empty.\n";
    help();
    return;
  }

  if (Arguments.front() == "find") {
    if (Arguments.size() != 2) {
      llvm::outs() << "find request must specify unqualified symbol name.\n";
      return;
    }
    reportTime("fuzzy find request",
               [&]() { fuzzyFind(Arguments.back(), Index); });
  } else if (Arguments.front() == "lookup") {
    if (Arguments.size() != 2) {
      llvm::outs() << "lookup request must specify symbol ID .\n";
      return;
    }
    reportTime("lookup request", [&]() { lookup(Arguments.back(), Index); });
  } else if (Arguments.front() == "help") {
    help();
  } else {
    llvm::outs() << "Unknown command. Try 'help'\n";
  }
}

} // namespace

int main(int argc, const char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  std::unique_ptr<SymbolIndex> Index;
  reportTime("Dex build", [&]() {
    Index = loadIndex(SymbolCollection, /*URISchemes=*/{},
                      /*UseDex=*/true);
  });

  if (!Index) {
    llvm::outs()
        << "ERROR: Please provide a valid path to symbol collection file.\n";
    return -1;
  }

  llvm::LineEditor LE("dexp");

  while (llvm::Optional<std::string> Request = LE.readLine())
    dispatch(Request.getValue(), *Index);

  return 0;
}
