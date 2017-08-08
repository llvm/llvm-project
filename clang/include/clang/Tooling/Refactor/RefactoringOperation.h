//===--- RefactoringOperations.h - Defines a refactoring operation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPERATION_H
#define LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPERATION_H

#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Refactor/RefactoringActions.h"
#include "clang/Tooling/Refactor/RefactoringOptionSet.h"
#include "clang/Tooling/Refactor/RefactoringReplacement.h"
#include "clang/Tooling/Refactor/SymbolOperation.h"
#include "llvm/ADT/None.h"
#include "llvm/Support/Error.h"
#include <string>
#include <vector>

namespace clang {

class ASTContext;
class Decl;
class Preprocessor;
class Stmt;

namespace tooling {

class RefactoringContinuation;

/// A refactoring result contains the source replacements produced by the
/// refactoring operation and the optional refactoring continuation.
struct RefactoringResult {
  std::vector<RefactoringReplacement> Replacements;
  std::vector<std::unique_ptr<RefactoringResultAssociatedSymbol>>
      AssociatedSymbols;
  std::unique_ptr<RefactoringContinuation> Continuation;

  RefactoringResult(
      std::vector<RefactoringReplacement> Replacements,
      std::unique_ptr<RefactoringContinuation> Continuation = nullptr)
      : Replacements(std::move(Replacements)),
        Continuation(std::move(Continuation)) {}

  RefactoringResult(std::unique_ptr<RefactoringContinuation> Continuation)
      : Replacements(), Continuation(std::move(Continuation)) {}

  RefactoringResult(RefactoringResult &&) = default;
  RefactoringResult &operator=(RefactoringResult &&) = default;
};

namespace indexer {

class IndexerQuery;
class ASTProducerQuery;

} // end namespace indexer

/// Refactoring continuations allow refactoring operations to run in external
/// AST units with some results that were obtained after querying the indexer.
///
/// The state of the refactoring operation is automatically managed by the
/// refactoring engine:
///   - Declaration references are converted to declaration references in
///     an external translation unit.
class RefactoringContinuation {
public:
  virtual ~RefactoringContinuation() {}

  virtual indexer::ASTProducerQuery *getASTUnitIndexerQuery() = 0;

  virtual std::vector<indexer::IndexerQuery *>
  getAdditionalIndexerQueries() = 0;

  /// Converts the TU-specific state in the continuation to a TU-independent
  /// state.
  ///
  /// This function is called before the initiation AST unit is freed.
  virtual void persistTUSpecificState() = 0;

  /// Invokes the continuation with the indexer query results and the state
  /// values in the context of another AST unit.
  virtual llvm::Expected<RefactoringResult>
  runInExternalASTUnit(ASTContext &Context) = 0;
};

// TODO: Remove in favour of diagnostics.
class RefactoringOperationError
    : public llvm::ErrorInfo<RefactoringOperationError> {
public:
  static char ID;
  StringRef FailureReason;

  RefactoringOperationError(StringRef FailureReason)
      : FailureReason(FailureReason) {}

  void log(raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override;
};

/// Represents an abstract refactoring operation.
class RefactoringOperation {
public:
  virtual ~RefactoringOperation() {}

  virtual const Stmt *getTransformedStmt() const { return nullptr; }

  virtual const Stmt *getLastTransformedStmt() const { return nullptr; }

  virtual const Decl *getTransformedDecl() const { return nullptr; }

  virtual const Decl *getLastTransformedDecl() const { return nullptr; }

  virtual std::vector<std::string> getRefactoringCandidates() { return {}; }

  virtual std::vector<RefactoringActionType> getAvailableSubActions() {
    return {};
  }

  virtual llvm::Expected<RefactoringResult>
  perform(ASTContext &Context, const Preprocessor &ThePreprocessor,
          const RefactoringOptionSet &Options,
          unsigned SelectedCandidateIndex = 0) = 0;
};

/// A wrapper around a unique pointer to a \c RefactoringOperation or \c
/// SymbolOperation that determines if the operation was successfully initiated
/// or not, even if the operation itself wasn't created.
struct RefactoringOperationResult {
  std::unique_ptr<RefactoringOperation> RefactoringOp;
  std::unique_ptr<SymbolOperation> SymbolOp;
  bool Initiated;
  StringRef FailureReason;

  RefactoringOperationResult() : Initiated(false) {}
  RefactoringOperationResult(llvm::NoneType) : Initiated(false) {}
  explicit RefactoringOperationResult(StringRef FailureReason)
      : Initiated(false), FailureReason(FailureReason) {}
};

/// Initiate a specific refactoring operation.
RefactoringOperationResult initiateRefactoringOperationAt(
    SourceLocation Location, SourceRange SelectionRange, ASTContext &Context,
    RefactoringActionType ActionType, bool CreateOperation = true);

/// Initiate a specific refactoring operation on a declaration that corresponds
/// to the given \p DeclUSR.
RefactoringOperationResult
initiateRefactoringOperationOnDecl(StringRef DeclUSR, ASTContext &Context,
                                   RefactoringActionType ActionType);

} // end namespace tooling
} // end namespace clang

#endif // LLVM_CLANG_TOOLING_REFACTOR_REFACTORING_OPERATION_H
