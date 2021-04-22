//===--- IndexerMain.cpp -----------------------------------------*- C++-*-===//
//
// TODO: Do things w/ license / header
//
//===----------------------------------------------------------------------===//
//
// clangd-indexer is a tool to gather index data (symbols, xrefs) from source.
//
//===----------------------------------------------------------------------===//

#include "index/IndexAction.h"
#include "index/LSIFSerialization.h"
#include "index/Merge.h"
#include "index/Ref.h"
#include "index/Serialization.h"
#include "index/Symbol.h"
#include "index/SymbolCollector.h"
#include "clang/Index/IndexingOptions.h"
#include "clang/Tooling/AllTUsExecution.h"
#include "clang/Tooling/ArgumentsAdjusters.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

#define BACKWARD_HAS_DWARF 1

using namespace clang::tooling;
using namespace llvm;

static cl::OptionCategory LSIFClangCategory("lsif-clang", R"(
  Creates an LSIF index of an entire project based on a JSON compilation database.

  $ lsif-clang compile_commands.json > dump.lsif
  )");

static cl::opt<std::string> ProjectRootArg(
    "project-root",
    cl::desc("Absolute path to root directory of project being indexed."),
    cl::init(""), cl::cat(LSIFClangCategory));

static cl::opt<std::string>
    IndexDestinationArg("out", cl::desc("Destination of resulting LSIF index."),
                        cl::init("dump.lsif"), cl::cat(LSIFClangCategory));

static cl::opt<bool> DebugArg("debug", cl::desc("Enable verbose debug output."),
                              cl::init(false), cl::cat(LSIFClangCategory));

static cl::opt<bool> DebugFilesArg("debug-files", cl::desc("Debug files being processed."),
                                   cl::init(false), cl::cat(LSIFClangCategory));

class IndexActionFactory : public FrontendActionFactory {
public:
  IndexActionFactory(clang::clangd::IndexFileIn &Result, std::string &ProjectRoot) : Result(Result), ProjectRoot(ProjectRoot) {}

  std::unique_ptr<clang::FrontendAction> create() override {
    clang::clangd::SymbolCollector::Options Opts;
    Opts.CountReferences = true;
    Opts.CollectMainFileSymbols = true;
    Opts.StoreAllDocumentation = true;
    clang::index::IndexingOptions IndexOpts;
    IndexOpts.IndexFunctionLocals = true;
    IndexOpts.IndexParametersInDeclarations = true;
    IndexOpts.IndexImplicitInstantiation = true;
    IndexOpts.IndexMacrosInPreprocessor = true;
    IndexOpts.SystemSymbolFilter =
        clang::index::IndexingOptions::SystemSymbolFilterKind::All;
    return createStaticIndexingAction(
        Opts, IndexOpts,
        [&](clang::clangd::SymbolSlab S) {
          // Merge as we go.
          std::lock_guard<std::mutex> Lock(SymbolsMu);
          for (const auto &Sym : S) {
            if (const auto *Existing = Symbols.find(Sym.ID))
              Symbols.insert(mergeSymbol(*Existing, Sym, ProjectRoot));
            else
              Symbols.insert(Sym);
          }
        },
        [&](clang::clangd::RefSlab S) {
          std::lock_guard<std::mutex> Lock(SymbolsMu);
          for (const auto &Sym : S) {
            // Deduplication happens during insertion.
            for (const auto &Ref : Sym.second)
              Refs.insert(Sym.first, Ref);
          }
        },
        [&](clang::clangd::RelationSlab S) {
          std::lock_guard<std::mutex> Lock(SymbolsMu);
          for (const auto &R : S) {
            Relations.insert(R);
          }
        },
        /*IncludeGraphCallback=*/nullptr);
  }

  // Awkward: we write the result in the destructor, because the executor
  // takes ownership so it's the easiest way to get our data back out.
  ~IndexActionFactory() {
    Result.Symbols = std::move(Symbols).build();
    Result.Refs = std::move(Refs).build();
    Result.Relations = std::move(Relations).build();
  }

private:
  clang::clangd::IndexFileIn &Result;
  std::mutex SymbolsMu;
  clang::clangd::SymbolSlab::Builder Symbols;
  clang::clangd::RefSlab::Builder Refs;
  clang::clangd::RelationSlab::Builder Relations;
  std::string &ProjectRoot;
};

int main(int argc, const char **argv) {
  //sys::PrintStackTraceOnErrorSignal(argv[0]);

  CommonOptionsParser OptionsParser(argc, argv, LSIFClangCategory,
                                    cl::OneOrMore);

  std::string ProjectRoot = ProjectRootArg;
  if (ProjectRoot == "") {
    SmallString<1024> CurrentPath;
    sys::fs::current_path(CurrentPath);
    ProjectRoot = std::string(CurrentPath.c_str());
  }
  ProjectRoot = clang::clangd::URI("file", "", ProjectRoot).toString();
  if (DebugArg) {
    llvm::errs() << "Using project root " << ProjectRoot << "\n";
  }

  ArgumentsAdjuster Adjuster = combineAdjusters(
      OptionsParser.getArgumentsAdjuster(),
      getInsertArgumentAdjuster("-Wno-unknown-warning-option"));

  // Collect symbols found in each translation unit, merging as we go.
  clang::clangd::IndexFileIn Data;

  auto& compilations = OptionsParser.getCompilations();
  if (compilations.getAllFiles().size() == 0) {
    exit(1);
  }
  AllTUsToolExecutor Executor(compilations, 0);
  auto Err =
    Executor.execute(std::make_unique<IndexActionFactory>(Data, ProjectRoot), Adjuster);
  if (Err) {
  }

  // Emit collected data.
  clang::clangd::IndexFileOut Out(Data);
  Out.Format = clang::clangd::IndexFileFormat::LSIF;
  Out.Debug = DebugArg;
  Out.DebugFiles = DebugFilesArg;
  Out.ProjectRoot = ProjectRoot;
  if (!IndexDestinationArg.empty()) {
    std::error_code FileErr;
    raw_fd_ostream IndexOstream(IndexDestinationArg, FileErr);
    if (FileErr.value() != 0) {
      report_fatal_error(FileErr.message());
    }
    writeLSIF(Out, IndexOstream);
  } else {
    writeLSIF(Out, outs());
  }
  return 0;
}
