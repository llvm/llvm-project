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
using namespace llvm;

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

// REPL commands inherit from Command and contain their options as members.
// Creating a Command populates parser options, parseAndRun() resets them.
class Command {
  // By resetting the parser options, we lost the standard -help flag.
  cl::opt<bool, false, cl::parser<bool>> Help{
      "help", cl::desc("Display available options"), cl::ValueDisallowed,
      cl::cat(cl::GeneralCategory)};
  virtual void run() = 0;

protected:
  const SymbolIndex *Index;

public:
  virtual ~Command() = default;
  virtual void parseAndRun(ArrayRef<const char *> Argv, const char *Overview,
                           const SymbolIndex &Index) {
    std::string ParseErrs;
    llvm::raw_string_ostream OS(ParseErrs);
    bool Ok =
        cl::ParseCommandLineOptions(Argv.size(), Argv.data(), Overview, &OS);
    if (Help.getNumOccurrences() > 0) {
      // Avoid printing parse errors in this case.
      // (Well, in theory. A bunch get printed to llvm::errs() regardless!)
      cl::PrintHelpMessage();
    } else {
      outs() << OS.str();
      if (Ok) {
        this->Index = &Index;
        reportTime(Argv[0], [&] { run(); });
      }
    }
    cl::ResetCommandLineParser(); // must do this before opts are destroyed.
  }
};

// FIXME(kbobyrev): Ideas for more commands:
// * find symbol references: print set of reference locations
// * load/swap/reload index: this would make it possible to get rid of llvm::cl
//   usages in the tool driver and actually use llvm::cl library in the REPL.
// * show posting list density histogram (our dump data somewhere so that user
//   could build one)
// * show number of tokens of each kind
// * print out tokens with the most dense posting lists
// * print out tokens with least dense posting lists

class FuzzyFind : public Command {
  cl::opt<std::string> Query{
      "query",
      cl::Positional,
      cl::Required,
      cl::desc("Query string to be fuzzy-matched"),
  };
  cl::opt<std::string> Scopes{
      "scopes",
      cl::desc("Allowed symbol scopes (comma-separated list)"),
  };
  cl::opt<unsigned> Limit{
      "limit",
      cl::init(10),
      cl::desc("Max results to display"),
  };

  void run() override {
    FuzzyFindRequest Request;
    Request.Limit = Limit;
    Request.Query = Query;
    if (Scopes.getNumOccurrences() > 0) {
      llvm::SmallVector<StringRef, 8> Scopes;
      StringRef(this->Scopes).split(Scopes, ',');
      Request.Scopes = {Scopes.begin(), Scopes.end()};
    }
    // FIXME(kbobyrev): Print symbol final scores to see the distribution.
    static const auto OutputFormat = "{0,-4} | {1,-40} | {2,-25}\n";
    llvm::outs() << llvm::formatv(OutputFormat, "Rank", "Symbol ID",
                                  "Symbol Name");
    size_t Rank = 0;
    Index->fuzzyFind(Request, [&](const Symbol &Sym) {
      llvm::outs() << llvm::formatv(OutputFormat, Rank++, Sym.ID.str(),
                                    Sym.Name);
    });
  }
};

class Lookup : public Command {
  cl::opt<std::string> ID{
      "id",
      cl::Positional,
      cl::Required,
      cl::desc("Symbol ID to look up (hex)"),
  };

  void run() override {
    auto SID = clang::clangd::SymbolID::fromStr(ID);
    if (!SID) {
      llvm::outs() << llvm::toString(SID.takeError()) << "\n";
      return;
    }

    clang::clangd::LookupRequest Request;
    Request.IDs = {*SID};
    bool FoundSymbol = false;
    Index->lookup(Request, [&](const Symbol &Sym) {
      FoundSymbol = true;
      llvm::outs() << SymbolToYAML(Sym);
    });
    if (!FoundSymbol)
      llvm::outs() << "not found\n";
  }
};

struct {
  const char *Name;
  const char *Description;
  std::function<std::unique_ptr<Command>()> Implementation;
} CommandInfo[] = {
    {"find", "Search for symbols with fuzzyFind", llvm::make_unique<FuzzyFind>},
    {"lookup", "Dump symbol details by ID", llvm::make_unique<Lookup>},
};

} // namespace

int main(int argc, const char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
  llvm::cl::ResetCommandLineParser(); // We reuse it for REPL commands.
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

  while (llvm::Optional<std::string> Request = LE.readLine()) {
    // Split on spaces and add required null-termination.
    std::replace(Request->begin(), Request->end(), ' ', '\0');
    SmallVector<StringRef, 8> Args;
    StringRef(*Request).split(Args, '\0', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    if (Args.empty())
      continue;
    if (Args.front() == "help") {
      outs() << "dexp - Index explorer\nCommands:\n";
      for (const auto &C : CommandInfo)
        outs() << llvm::formatv("{0,16} - {1}\n", C.Name, C.Description);
      outs() << "Get detailed command help with e.g. `find -help`.\n";
      continue;
    }
    SmallVector<const char *, 8> FakeArgv;
    for (StringRef S : Args)
      FakeArgv.push_back(S.data()); // Terminated by separator or end of string.

    bool Recognized = false;
    for (const auto &Cmd : CommandInfo) {
      if (Cmd.Name == Args.front()) {
        Recognized = true;
        Cmd.Implementation()->parseAndRun(FakeArgv, Cmd.Description, *Index);
        break;
      }
    }
    if (!Recognized)
      outs() << "Unknown command. Try 'help'.\n";
  }

  return 0;
}
