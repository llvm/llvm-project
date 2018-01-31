#include "TUScheduler.h"
#include "clang/Frontend/PCHContainerOperations.h"
#include "llvm/Support/Errc.h"

namespace clang {
namespace clangd {
unsigned getDefaultAsyncThreadsCount() {
  unsigned HardwareConcurrency = std::thread::hardware_concurrency();
  // C++ standard says that hardware_concurrency()
  // may return 0, fallback to 1 worker thread in
  // that case.
  if (HardwareConcurrency == 0)
    return 1;
  return HardwareConcurrency;
}

TUScheduler::TUScheduler(unsigned AsyncThreadsCount,
                         bool StorePreamblesInMemory,
                         ASTParsedCallback ASTCallback)

    : Files(StorePreamblesInMemory, std::make_shared<PCHContainerOperations>(),
            std::move(ASTCallback)),
      Threads(AsyncThreadsCount) {}

void TUScheduler::update(
    PathRef File, ParseInputs Inputs,
    UniqueFunction<void(llvm::Optional<std::vector<DiagWithFixIts>>)>
        OnUpdated) {
  CachedInputs[File] = Inputs;

  auto Resources = Files.getOrCreateFile(File);
  auto DeferredRebuild = Resources->deferRebuild(std::move(Inputs));

  Threads.addToFront(
      [](decltype(OnUpdated) OnUpdated,
         decltype(DeferredRebuild) DeferredRebuild) {
        auto Diags = DeferredRebuild();
        OnUpdated(Diags);
      },
      std::move(OnUpdated), std::move(DeferredRebuild));
}

void TUScheduler::remove(PathRef File,
                         UniqueFunction<void(llvm::Error)> Action) {
  CachedInputs.erase(File);

  auto Resources = Files.removeIfPresent(File);
  if (!Resources) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to remove non-added document", llvm::errc::invalid_argument));
    return;
  }

  auto DeferredCancel = Resources->deferCancelRebuild();
  Threads.addToFront(
      [](decltype(Action) Action, decltype(DeferredCancel) DeferredCancel) {
        DeferredCancel();
        Action(llvm::Error::success());
      },
      std::move(Action), std::move(DeferredCancel));
}

void TUScheduler::runWithAST(
    PathRef File, UniqueFunction<void(llvm::Expected<InputsAndAST>)> Action) {
  auto Resources = Files.getFile(File);
  if (!Resources) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to get AST for non-added document",
        llvm::errc::invalid_argument));
    return;
  }

  const ParseInputs &Inputs = getInputs(File);
  // We currently block the calling thread until AST is available and run the
  // action on the calling thread to avoid inconsistent states coming from
  // subsequent updates.
  // FIXME(ibiryukov): this should be moved to the worker threads.
  Resources->getAST().get()->runUnderLock([&](ParsedAST *AST) {
    if (AST)
      Action(InputsAndAST{Inputs, *AST});
    else
      Action(llvm::make_error<llvm::StringError>(
          "Could not build AST for the latest file update",
          llvm::errc::invalid_argument));
  });
}

void TUScheduler::runWithPreamble(
    PathRef File,
    UniqueFunction<void(llvm::Expected<InputsAndPreamble>)> Action) {
  std::shared_ptr<CppFile> Resources = Files.getFile(File);
  if (!Resources) {
    Action(llvm::make_error<llvm::StringError>(
        "trying to get preamble for non-added document",
        llvm::errc::invalid_argument));
    return;
  }

  const ParseInputs &Inputs = getInputs(File);
  std::shared_ptr<const PreambleData> Preamble =
      Resources->getPossiblyStalePreamble();
  Threads.addToFront(
      [Resources, Preamble, Inputs](decltype(Action) Action) mutable {
        if (!Preamble)
          Preamble = Resources->getPossiblyStalePreamble();

        Action(InputsAndPreamble{Inputs, Preamble.get()});
      },
      std::move(Action));
}

const ParseInputs &TUScheduler::getInputs(PathRef File) {
  auto It = CachedInputs.find(File);
  assert(It != CachedInputs.end());
  return It->second;
}

std::vector<std::pair<Path, std::size_t>>
TUScheduler::getUsedBytesPerFile() const {
  return Files.getUsedBytesPerFile();
}
} // namespace clangd
} // namespace clang
