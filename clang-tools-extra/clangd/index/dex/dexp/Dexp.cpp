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

#include "SourceCode.h"
#include "index/Serialization.h"
#include "index/dex/Dex.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/LineEditor/LineEditor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Signals.h"

namespace clang {
namespace clangd {
namespace {

llvm::cl::opt<std::string> IndexPath("index-path",
                                     llvm::cl::desc("Path to the index"),
                                     llvm::cl::Positional, llvm::cl::Required);

static const std::string Overview = R"(
This is an **experimental** interactive tool to process user-provided search
queries over given symbol collection obtained via clangd-indexer. The
tool can be used to evaluate search quality of existing index implementations
and manually construct non-trivial test cases.

Type use "help" request to get information about the details.
)";

void reportTime(llvm::StringRef Name, llvm::function_ref<void()> F) {
  const auto TimerStart = std::chrono::high_resolution_clock::now();
  F();
  const auto TimerStop = std::chrono::high_resolution_clock::now();
  const auto Duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      TimerStop - TimerStart);
  llvm::outs() << llvm::formatv("{0} took {1:ms+n}.\n", Name, Duration);
}

std::vector<SymbolID> getSymbolIDsFromIndex(llvm::StringRef QualifiedName,
                                            const SymbolIndex *Index) {
  FuzzyFindRequest Request;
  // Remove leading "::" qualifier as FuzzyFind doesn't need leading "::"
  // qualifier for global scope.
  bool IsGlobalScope = QualifiedName.consume_front("::");
  auto Names = splitQualifiedName(QualifiedName);
  if (IsGlobalScope || !Names.first.empty())
    Request.Scopes = {Names.first};
  else
    // QualifiedName refers to a symbol in global scope (e.g. "GlobalSymbol"),
    // add the global scope to the request.
    Request.Scopes = {""};

  Request.Query = Names.second;
  std::vector<SymbolID> SymIDs;
  Index->fuzzyFind(Request, [&](const Symbol &Sym) {
    std::string SymQualifiedName = (Sym.Scope + Sym.Name).str();
    if (QualifiedName == SymQualifiedName)
      SymIDs.push_back(Sym.ID);
  });
  return SymIDs;
}

// REPL commands inherit from Command and contain their options as members.
// Creating a Command populates parser options, parseAndRun() resets them.
class Command {
  // By resetting the parser options, we lost the standard -help flag.
  llvm::cl::opt<bool, false, llvm::cl::parser<bool>> Help{
      "help", llvm::cl::desc("Display available options"),
      llvm::cl::ValueDisallowed, llvm::cl::cat(llvm::cl::GeneralCategory)};
  virtual void run() = 0;

protected:
  const SymbolIndex *Index;

public:
  virtual ~Command() = default;
  virtual void parseAndRun(llvm::ArrayRef<const char *> Argv,
                           const char *Overview, const SymbolIndex &Index) {
    std::string ParseErrs;
    llvm::raw_string_ostream OS(ParseErrs);
    bool Ok = llvm::cl::ParseCommandLineOptions(Argv.size(), Argv.data(),
                                                Overview, &OS);
    if (Help.getNumOccurrences() > 0) {
      // Avoid printing parse errors in this case.
      // (Well, in theory. A bunch get printed to llvm::errs() regardless!)
      llvm::cl::PrintHelpMessage();
    } else {
      llvm::outs() << OS.str();
      if (Ok) {
        this->Index = &Index;
        reportTime(Argv[0], [&] { run(); });
      }
    }
    llvm::cl::ResetCommandLineParser(); // must do this before opts are
                                        // destroyed.
  }
};

// FIXME(kbobyrev): Ideas for more commands:
// * load/swap/reload index: this would make it possible to get rid of llvm::cl
//   usages in the tool driver and actually use llvm::cl library in the REPL.
// * show posting list density histogram (our dump data somewhere so that user
//   could build one)
// * show number of tokens of each kind
// * print out tokens with the most dense posting lists
// * print out tokens with least dense posting lists

class FuzzyFind : public Command {
  llvm::cl::opt<std::string> Query{
      "query",
      llvm::cl::Positional,
      llvm::cl::Required,
      llvm::cl::desc("Query string to be fuzzy-matched"),
  };
  llvm::cl::opt<std::string> Scopes{
      "scopes",
      llvm::cl::desc("Allowed symbol scopes (comma-separated list)"),
  };
  llvm::cl::opt<unsigned> Limit{
      "limit",
      llvm::cl::init(10),
      llvm::cl::desc("Max results to display"),
  };

  void run() override {
    FuzzyFindRequest Request;
    Request.Limit = Limit;
    Request.Query = Query;
    if (Scopes.getNumOccurrences() > 0) {
      llvm::SmallVector<llvm::StringRef, 8> Scopes;
      llvm::StringRef(this->Scopes).split(Scopes, ',');
      Request.Scopes = {Scopes.begin(), Scopes.end()};
    }
    Request.AnyScope = Request.Scopes.empty();
    // FIXME(kbobyrev): Print symbol final scores to see the distribution.
    static const auto OutputFormat = "{0,-4} | {1,-40} | {2,-25}\n";
    llvm::outs() << llvm::formatv(OutputFormat, "Rank", "Symbol ID",
                                  "Symbol Name");
    size_t Rank = 0;
    Index->fuzzyFind(Request, [&](const Symbol &Sym) {
      llvm::outs() << llvm::formatv(OutputFormat, Rank++, Sym.ID.str(),
                                    Sym.Scope + Sym.Name);
    });
  }
};

class Lookup : public Command {
  llvm::cl::opt<std::string> ID{
      "id",
      llvm::cl::Positional,
      llvm::cl::desc("Symbol ID to look up (hex)"),
  };
  llvm::cl::opt<std::string> Name{
      "name",
      llvm::cl::desc("Qualified name to look up."),
  };

  void run() override {
    if (ID.getNumOccurrences() == 0 && Name.getNumOccurrences() == 0) {
      llvm::outs()
          << "Missing required argument: please provide id or -name.\n";
      return;
    }
    std::vector<SymbolID> IDs;
    if (ID.getNumOccurrences()) {
      auto SID = SymbolID::fromStr(ID);
      if (!SID) {
        llvm::outs() << llvm::toString(SID.takeError()) << "\n";
        return;
      }
      IDs.push_back(*SID);
    } else {
      IDs = getSymbolIDsFromIndex(Name, Index);
    }

    LookupRequest Request;
    Request.IDs.insert(IDs.begin(), IDs.end());
    bool FoundSymbol = false;
    Index->lookup(Request, [&](const Symbol &Sym) {
      FoundSymbol = true;
      llvm::outs() << toYAML(Sym);
    });
    if (!FoundSymbol)
      llvm::outs() << "not found\n";
  }
};

class Refs : public Command {
  llvm::cl::opt<std::string> ID{
      "id",
      llvm::cl::Positional,
      llvm::cl::desc("Symbol ID of the symbol being queried (hex)."),
  };
  llvm::cl::opt<std::string> Name{
      "name",
      llvm::cl::desc("Qualified name of the symbol being queried."),
  };
  llvm::cl::opt<std::string> Filter{
      "filter",
      llvm::cl::init(".*"),
      llvm::cl::desc(
          "Print all results from files matching this regular expression."),
  };

  void run() override {
    if (ID.getNumOccurrences() == 0 && Name.getNumOccurrences() == 0) {
      llvm::outs()
          << "Missing required argument: please provide id or -name.\n";
      return;
    }
    std::vector<SymbolID> IDs;
    if (ID.getNumOccurrences()) {
      auto SID = SymbolID::fromStr(ID);
      if (!SID) {
        llvm::outs() << llvm::toString(SID.takeError()) << "\n";
        return;
      }
      IDs.push_back(*SID);
    } else {
      IDs = getSymbolIDsFromIndex(Name, Index);
      if (IDs.size() > 1) {
        llvm::outs() << llvm::formatv(
            "The name {0} is ambiguous, found {1} different "
            "symbols. Please use id flag to disambiguate.\n",
            Name, IDs.size());
        return;
      }
    }
    RefsRequest RefRequest;
    RefRequest.IDs.insert(IDs.begin(), IDs.end());
    llvm::Regex RegexFilter(Filter);
    Index->refs(RefRequest, [&RegexFilter](const Ref &R) {
      auto U = URI::parse(R.Location.FileURI);
      if (!U) {
        llvm::outs() << U.takeError();
        return;
      }
      if (RegexFilter.match(U->body()))
        llvm::outs() << R << "\n";
    });
  }
};

struct {
  const char *Name;
  const char *Description;
  std::function<std::unique_ptr<Command>()> Implementation;
} CommandInfo[] = {
    {"find", "Search for symbols with fuzzyFind", llvm::make_unique<FuzzyFind>},
    {"lookup", "Dump symbol details by ID or qualified name",
     llvm::make_unique<Lookup>},
    {"refs", "Find references by ID or qualified name",
     llvm::make_unique<Refs>},
};

std::unique_ptr<SymbolIndex> openIndex(llvm::StringRef Index) {
  return loadIndex(Index, /*UseDex=*/true);
}

} // namespace
} // namespace clangd
} // namespace clang

int main(int argc, const char *argv[]) {
  using namespace clang::clangd;

  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
  llvm::cl::ResetCommandLineParser(); // We reuse it for REPL commands.
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  std::unique_ptr<SymbolIndex> Index;
  reportTime("Dex build", [&]() {
    Index = openIndex(IndexPath);
  });

  if (!Index) {
    llvm::outs() << "Failed to open the index.\n";
    return -1;
  }

  llvm::LineEditor LE("dexp");

  while (llvm::Optional<std::string> Request = LE.readLine()) {
    // Split on spaces and add required null-termination.
    std::replace(Request->begin(), Request->end(), ' ', '\0');
    llvm::SmallVector<llvm::StringRef, 8> Args;
    llvm::StringRef(*Request).split(Args, '\0', /*MaxSplit=*/-1,
                                    /*KeepEmpty=*/false);
    if (Args.empty())
      continue;
    if (Args.front() == "help") {
      llvm::outs() << "dexp - Index explorer\nCommands:\n";
      for (const auto &C : CommandInfo)
        llvm::outs() << llvm::formatv("{0,16} - {1}\n", C.Name, C.Description);
      llvm::outs() << "Get detailed command help with e.g. `find -help`.\n";
      continue;
    }
    llvm::SmallVector<const char *, 8> FakeArgv;
    for (llvm::StringRef S : Args)
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
      llvm::outs() << "Unknown command. Try 'help'.\n";
  }

  return 0;
}
