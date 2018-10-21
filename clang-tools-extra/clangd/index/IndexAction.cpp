#include "IndexAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Index/IndexDataConsumer.h"
#include "clang/Index/IndexingAction.h"
#include "clang/Tooling/Tooling.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace {

// Wraps the index action and reports index data after each translation unit.
class IndexAction : public WrapperFrontendAction {
public:
  IndexAction(std::shared_ptr<SymbolCollector> C,
              std::unique_ptr<CanonicalIncludes> Includes,
              const index::IndexingOptions &Opts,
              std::function<void(SymbolSlab)> SymbolsCallback,
              std::function<void(RefSlab)> RefsCallback)
      : WrapperFrontendAction(index::createIndexingAction(C, Opts, nullptr)),
        SymbolsCallback(SymbolsCallback), RefsCallback(RefsCallback),
        Collector(C), Includes(std::move(Includes)),
        PragmaHandler(collectIWYUHeaderMaps(this->Includes.get())) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef InFile) override {
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
      errs() << "Skipping TU due to uncompilable errors\n";
      return;
    }
    SymbolsCallback(Collector->takeSymbols());
    if (RefsCallback != nullptr)
      RefsCallback(Collector->takeRefs());
  }

private:
  std::function<void(SymbolSlab)> SymbolsCallback;
  std::function<void(RefSlab)> RefsCallback;
  std::shared_ptr<SymbolCollector> Collector;
  std::unique_ptr<CanonicalIncludes> Includes;
  std::unique_ptr<CommentHandler> PragmaHandler;
};

} // namespace

std::unique_ptr<FrontendAction>
createStaticIndexingAction(SymbolCollector::Options Opts,
                           std::function<void(SymbolSlab)> SymbolsCallback,
                           std::function<void(RefSlab)> RefsCallback) {
  index::IndexingOptions IndexOpts;
  IndexOpts.SystemSymbolFilter =
      index::IndexingOptions::SystemSymbolFilterKind::All;
  Opts.CollectIncludePath = true;
  Opts.CountReferences = true;
  Opts.Origin = SymbolOrigin::Static;
  if (RefsCallback != nullptr) {
    Opts.RefFilter = RefKind::All;
    Opts.RefsInHeaders = true;
  }
  auto Includes = llvm::make_unique<CanonicalIncludes>();
  addSystemHeadersMapping(Includes.get());
  Opts.Includes = Includes.get();
  return llvm::make_unique<IndexAction>(
      std::make_shared<SymbolCollector>(std::move(Opts)), std::move(Includes),
      IndexOpts, SymbolsCallback, RefsCallback);
}

} // namespace clangd
} // namespace clang
