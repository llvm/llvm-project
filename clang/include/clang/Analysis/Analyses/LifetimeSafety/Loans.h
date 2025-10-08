#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
#include "LifetimeSafety.h"

namespace clang::lifetimes {
namespace internal {

/// Represents the storage location being borrowed, e.g., a specific stack
/// variable.
/// TODO: Model access paths of other types, e.g., s.field, heap and globals.
struct AccessPath {
  const clang::ValueDecl *D;

  AccessPath(const clang::ValueDecl *D) : D(D) {}
};

/// Information about a single borrow, or "Loan". A loan is created when a
/// reference or pointer is created.
struct Loan {
  /// TODO: Represent opaque loans.
  /// TODO: Represent nullptr: loans to no path. Accessing it UB! Currently it
  /// is represented as empty LoanSet
  LoanID ID;
  AccessPath Path;
  /// The expression that creates the loan, e.g., &x.
  const Expr *IssueExpr;

  Loan(LoanID id, AccessPath path, const Expr *IssueExpr)
      : ID(id), Path(path), IssueExpr(IssueExpr) {}

  void dump(llvm::raw_ostream &OS) const {
    OS << ID << " (Path: ";
    OS << Path.D->getNameAsString() << ")";
  }
};

/// Manages the creation, storage and retrieval of loans.
class LoanManager {
public:
  LoanManager() = default;

  Loan &addLoan(AccessPath Path, const Expr *IssueExpr) {
    AllLoans.emplace_back(getNextLoanID(), Path, IssueExpr);
    return AllLoans.back();
  }

  const Loan &getLoan(LoanID ID) const {
    assert(ID.Value < AllLoans.size());
    return AllLoans[ID.Value];
  }
  llvm::ArrayRef<Loan> getLoans() const { return AllLoans; }

private:
  LoanID getNextLoanID() { return NextLoanID++; }

  LoanID NextLoanID{0};
  /// TODO(opt): Profile and evaluate the usefullness of small buffer
  /// optimisation.
  llvm::SmallVector<Loan> AllLoans;
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOANS_H
