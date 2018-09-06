//===--- GlobalSymbolBuilderMain.cpp -----------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GlobalSymbolBuilder is a tool to extract symbols from a whole project.
// This tool is **experimental** only. Don't use it in production code.
//
//===----------------------------------------------------------------------===//

#include "RIFF.h"
#include "index/CanonicalIncludes.h"
#include "index/Index.h"
#include "index/Merge.h"
#include "index/Serialization.h"
#include "index/SymbolCollector.h"
#include "index/SymbolYAML.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;
using namespace clang::tooling;
using clang::clangd::SymbolSlab;

namespace clang {
namespace clangd {
namespace {

static llvm::cl::opt<std::string> AssumedHeaderDir(
    "assume-header-dir",
    llvm::cl::desc("The index includes header that a symbol is defined in. "
                   "If the absolute path cannot be determined (e.g. an "
                   "in-memory VFS) then the relative path is resolved against "
                   "this directory, which must be absolute. If this flag is "
                   "not given, such headers will have relative paths."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> MergeOnTheFly(
    "merge-on-the-fly",
    llvm::cl::desc(
        "Merges symbols for each processed translation unit as soon "
        "they become available. This results in a smaller memory "
        "usage and an almost instant reduce stage. Optimal for running as a "
        "standalone tool, but cannot be used with multi-process executors like "
        "MapReduce."),
    llvm::cl::init(true), llvm::cl::Hidden);

enum IndexFormat { YAML, Binary };
static llvm::cl::opt<IndexFormat> Format(
    "format", llvm::cl::desc("Format of the index to be written"),
    llvm::cl::values(clEnumValN(YAML, "yaml", "human-readable YAML format"),
                     clEnumValN(Binary, "binary", "binary RIFF format")),
    llvm::cl::init(YAML));

/// Responsible for aggregating symbols from each processed file and producing
/// the final results. All methods in this class must be thread-safe,
/// 'consumeSymbols' may be called from multiple threads.
class SymbolsConsumer {
public:
  virtual ~SymbolsConsumer() = default;

  /// Consume a SymbolSlab build for a file.
  virtual void consumeSymbols(SymbolSlab Symbols) = 0;
  /// Produce a resulting symbol slab, by combining  occurrences of the same
  /// symbols across translation units.
  virtual SymbolSlab mergeResults() = 0;
};

class SymbolIndexActionFactory : public tooling::FrontendActionFactory {
public:
  SymbolIndexActionFactory(SymbolsConsumer &Consumer) : Consumer(Consumer) {}

  clang::FrontendAction *create() override {
    // Wraps the index action and reports collected symbols to the execution
    // context at the end of each translation unit.
    class WrappedIndexAction : public WrapperFrontendAction {
    public:
      WrappedIndexAction(std::shared_ptr<SymbolCollector> C,
                         std::unique_ptr<CanonicalIncludes> Includes,
                         const index::IndexingOptions &Opts,
                         SymbolsConsumer &Consumer)
          : WrapperFrontendAction(
                index::createIndexingAction(C, Opts, nullptr)),
            Consumer(Consumer), Collector(C), Includes(std::move(Includes)),
            PragmaHandler(collectIWYUHeaderMaps(this->Includes.get())) {}

      std::unique_ptr<ASTConsumer>
      CreateASTConsumer(CompilerInstance &CI, StringRef InFile) override {
        CI.getPreprocessor().addCommentHandler(PragmaHandler.get());
        return WrapperFrontendAction::CreateASTConsumer(CI, InFile);
      }

      bool BeginInvocation(CompilerInstance &CI) override {
        // We want all comments, not just the doxygen ones.
        CI.getLangOpts().CommentOpts.ParseAllComments = true;
        return WrapperFrontendAction::BeginInvocation(CI);
      }

      void EndSourceFileAction() override {
        WrapperFrontendAction::EndSourceFileAction();

        const auto &CI = getCompilerInstance();
        if (CI.hasDiagnostics() &&
            CI.getDiagnostics().hasUncompilableErrorOccurred()) {
          llvm::errs()
              << "Found uncompilable errors in the translation unit. Igoring "
                 "collected symbols...\n";
          return;
        }

        Consumer.consumeSymbols(Collector->takeSymbols());
      }

    private:
      SymbolsConsumer &Consumer;
      std::shared_ptr<SymbolCollector> Collector;
      std::unique_ptr<CanonicalIncludes> Includes;
      std::unique_ptr<CommentHandler> PragmaHandler;
    };

    index::IndexingOptions IndexOpts;
    IndexOpts.SystemSymbolFilter =
        index::IndexingOptions::SystemSymbolFilterKind::All;
    IndexOpts.IndexFunctionLocals = false;
    auto CollectorOpts = SymbolCollector::Options();
    CollectorOpts.FallbackDir = AssumedHeaderDir;
    CollectorOpts.CollectIncludePath = true;
    CollectorOpts.CountReferences = true;
    CollectorOpts.Origin = SymbolOrigin::Static;
    auto Includes = llvm::make_unique<CanonicalIncludes>();
    addSystemHeadersMapping(Includes.get());
    CollectorOpts.Includes = Includes.get();
    return new WrappedIndexAction(
        std::make_shared<SymbolCollector>(std::move(CollectorOpts)),
        std::move(Includes), IndexOpts, Consumer);
  }

  SymbolsConsumer &Consumer;
};

/// Stashes per-file results inside ExecutionContext, merges all of them at the
/// end. Useful for running on MapReduce infrastructure to avoid keeping symbols
/// from multiple files in memory.
class ToolExecutorConsumer : public SymbolsConsumer {
public:
  ToolExecutorConsumer(ToolExecutor &Executor) : Executor(Executor) {}

  void consumeSymbols(SymbolSlab Symbols) override {
    for (const auto &Sym : Symbols)
      Executor.getExecutionContext()->reportResult(Sym.ID.str(),
                                                   SymbolToYAML(Sym));
  }

  SymbolSlab mergeResults() override {
    SymbolSlab::Builder UniqueSymbols;
    Executor.getToolResults()->forEachResult(
        [&](llvm::StringRef Key, llvm::StringRef Value) {
          llvm::yaml::Input Yin(Value);
          auto Sym = clang::clangd::SymbolFromYAML(Yin);
          clang::clangd::SymbolID ID;
          Key >> ID;
          if (const auto *Existing = UniqueSymbols.find(ID))
            UniqueSymbols.insert(mergeSymbol(*Existing, Sym));
          else
            UniqueSymbols.insert(Sym);
        });
    return std::move(UniqueSymbols).build();
  }

private:
  ToolExecutor &Executor;
};

/// Merges symbols for each translation unit as soon as the file is processed.
/// Optimal choice for standalone tools.
class OnTheFlyConsumer : public SymbolsConsumer {
public:
  void consumeSymbols(SymbolSlab Symbols) override {
    std::lock_guard<std::mutex> Lock(Mut);
    for (auto &&Sym : Symbols) {
      if (const auto *Existing = Result.find(Sym.ID))
        Result.insert(mergeSymbol(*Existing, Sym));
      else
        Result.insert(Sym);
    }
  }

  SymbolSlab mergeResults() override {
    std::lock_guard<std::mutex> Lock(Mut);
    return std::move(Result).build();
  }

private:
  std::mutex Mut;
  SymbolSlab::Builder Result;
};

} // namespace
} // namespace clangd
} // namespace clang

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  const char *Overview = R"(
  This is an **experimental** tool to extract symbols from a whole project
  for clangd (global code completion). It will be changed and deprecated
  eventually. Don't use it in production code!

  Example usage for building index for the whole project using CMake compile
  commands:

  $ global-symbol-builder --executor=all-TUs compile_commands.json > index.yaml

  Example usage for file sequence index without flags:

  $ global-symbol-builder File1.cpp File2.cpp ... FileN.cpp > index.yaml

  Note: only symbols from header files will be collected.
  )";

  auto Executor = clang::tooling::createExecutorFromCommandLineArgs(
      argc, argv, cl::GeneralCategory, Overview);

  if (!Executor) {
    llvm::errs() << llvm::toString(Executor.takeError()) << "\n";
    return 1;
  }

  if (!clang::clangd::AssumedHeaderDir.empty() &&
      !llvm::sys::path::is_absolute(clang::clangd::AssumedHeaderDir)) {
    llvm::errs() << "--assume-header-dir must be an absolute path.\n";
    return 1;
  }

  if (clang::clangd::MergeOnTheFly && !Executor->get()->isSingleProcess()) {
    llvm::errs()
        << "Found multi-process executor, forcing the use of intermediate YAML "
           "serialization instead of the on-the-fly merge.\n";
    clang::clangd::MergeOnTheFly = false;
  }

  std::unique_ptr<clang::clangd::SymbolsConsumer> Consumer;
  if (clang::clangd::MergeOnTheFly)
    Consumer = llvm::make_unique<clang::clangd::OnTheFlyConsumer>();
  else
    Consumer =
        llvm::make_unique<clang::clangd::ToolExecutorConsumer>(**Executor);

  // Map phase: emit symbols found in each translation unit.
  auto Err = Executor->get()->execute(
      llvm::make_unique<clang::clangd::SymbolIndexActionFactory>(*Consumer));
  if (Err) {
    llvm::errs() << llvm::toString(std::move(Err)) << "\n";
  }
  // Reduce phase: combine symbols with the same IDs.
  auto UniqueSymbols = Consumer->mergeResults();
  // Output phase: emit result symbols.
  switch (clang::clangd::Format) {
  case clang::clangd::IndexFormat::YAML:
    SymbolsToYAML(UniqueSymbols, llvm::outs());
    break;
  case clang::clangd::IndexFormat::Binary: {
    clang::clangd::IndexFileOut Out;
    Out.Symbols = &UniqueSymbols;
    llvm::outs() << Out;
  }
  }
  return 0;
}
