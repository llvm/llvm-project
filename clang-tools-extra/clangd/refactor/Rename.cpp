#include "refactor/Rename.h"
#include "clang/Tooling/Refactoring/RefactoringResultConsumer.h"
#include "clang/Tooling/Refactoring/Rename/RenamingAction.h"

namespace clang {
namespace clangd {
namespace {

class RefactoringResultCollector final
    : public tooling::RefactoringResultConsumer {
public:
  void handleError(llvm::Error Err) override {
    assert(!Result.hasValue());
    Result = std::move(Err);
  }

  // Using the handle(SymbolOccurrences) from parent class.
  using tooling::RefactoringResultConsumer::handle;

  void handle(tooling::AtomicChanges SourceReplacements) override {
    assert(!Result.hasValue());
    Result = std::move(SourceReplacements);
  }

  llvm::Optional<llvm::Expected<tooling::AtomicChanges>> Result;
};

// Expand a DiagnosticError to make it print-friendly (print the detailed
// message, rather than "clang diagnostic").
llvm::Error expandDiagnostics(llvm::Error Err, DiagnosticsEngine &DE) {
  if (auto Diag = DiagnosticError::take(Err)) {
    llvm::cantFail(std::move(Err));
    SmallVector<char, 128> DiagMessage;
    Diag->second.EmitToString(DE, DiagMessage);
    return llvm::make_error<llvm::StringError>(DiagMessage,
                                               llvm::inconvertibleErrorCode());
  }
  return Err;
}

} // namespace

llvm::Expected<tooling::Replacements>
renameWithinFile(ParsedAST &AST, llvm::StringRef File, Position Pos,
                 llvm::StringRef NewName) {
  RefactoringResultCollector ResultCollector;
  ASTContext &ASTCtx = AST.getASTContext();
  const SourceManager &SourceMgr = ASTCtx.getSourceManager();
  SourceLocation SourceLocationBeg =
      clangd::getBeginningOfIdentifier(AST, Pos, SourceMgr.getMainFileID());
  tooling::RefactoringRuleContext Context(ASTCtx.getSourceManager());
  Context.setASTContext(ASTCtx);
  auto Rename = clang::tooling::RenameOccurrences::initiate(
      Context, SourceRange(SourceLocationBeg), NewName);
  if (!Rename)
    return expandDiagnostics(Rename.takeError(), ASTCtx.getDiagnostics());

  Rename->invoke(ResultCollector, Context);

  assert(ResultCollector.Result.hasValue());
  if (!ResultCollector.Result.getValue())
    return expandDiagnostics(ResultCollector.Result->takeError(),
                             ASTCtx.getDiagnostics());

  tooling::Replacements FilteredChanges;
  // Right now we only support renaming the main file, so we
  // drop replacements not for the main file. In the future, we might
  // also support rename with wider scope.
  // Rename sometimes returns duplicate edits (which is a bug). A side-effect of 
  // adding them to a single Replacements object is these are deduplicated.
  for (const tooling::AtomicChange &Change : ResultCollector.Result->get()) {
    for (const auto &Rep : Change.getReplacements()) {
      if (Rep.getFilePath() == File)
        cantFail(FilteredChanges.add(Rep));
    }
  }
  return FilteredChanges;
}

} // namespace clangd
} // namespace clang
