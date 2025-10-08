#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H

#include "clang/Analysis/Analyses/LifetimeSafety/Dataflow.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeSafety.h"
#include "clang/Analysis/Analyses/LifetimeSafety/Utils.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Analysis/CFG.h"
#include "llvm/ADT/ImmutableMap.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/Support/Debug.h"

namespace clang::lifetimes {
namespace internal {

// ========================================================================= //
//                          Loan Propagation Analysis
// ========================================================================= //

/// Represents the dataflow lattice for loan propagation.
///
/// This lattice tracks which loans each origin may hold at a given program
/// point.The lattice has a finite height: An origin's loan set is bounded by
/// the total number of loans in the function.
/// TODO(opt): To reduce the lattice size, propagate origins of declarations,
/// not expressions, because expressions are not visible across blocks.
struct LoanPropagationLattice {
  /// The map from an origin to the set of loans it contains.
  OriginLoanMap Origins = OriginLoanMap(nullptr);

  explicit LoanPropagationLattice(const OriginLoanMap &S) : Origins(S) {}
  LoanPropagationLattice() = default;

  bool operator==(const LoanPropagationLattice &Other) const {
    return Origins == Other.Origins;
  }
  bool operator!=(const LoanPropagationLattice &Other) const {
    return !(*this == Other);
  }

  void dump(llvm::raw_ostream &OS) const {
    OS << "LoanPropagationLattice State:\n";
    if (Origins.isEmpty())
      OS << "  <empty>\n";
    for (const auto &Entry : Origins) {
      if (Entry.second.isEmpty())
        OS << "  Origin " << Entry.first << " contains no loans\n";
      for (const LoanID &LID : Entry.second)
        OS << "  Origin " << Entry.first << " contains Loan " << LID << "\n";
    }
  }
};

/// The analysis that tracks which loans belong to which origins.
class LoanPropagationAnalysis
    : public DataflowAnalysis<LoanPropagationAnalysis, LoanPropagationLattice,
                              Direction::Forward> {
  OriginLoanMap::Factory &OriginLoanMapFactory;
  LoanSet::Factory &LoanSetFactory;

public:
  LoanPropagationAnalysis(const CFG &C, AnalysisDeclContext &AC, FactManager &F,
                          OriginLoanMap::Factory &OriginLoanMapFactory,
                          LoanSet::Factory &LoanSetFactory)
      : DataflowAnalysis(C, AC, F), OriginLoanMapFactory(OriginLoanMapFactory),
        LoanSetFactory(LoanSetFactory) {}

  using Base::transfer;

  StringRef getAnalysisName() const { return "LoanPropagation"; }

  Lattice getInitialState() { return Lattice{}; }

  /// Merges two lattices by taking the union of loans for each origin.
  // TODO(opt): Keep the state small by removing origins which become dead.
  Lattice join(Lattice A, Lattice B) {
    OriginLoanMap JoinedOrigins = utils::join(
        A.Origins, B.Origins, OriginLoanMapFactory,
        [&](const LoanSet *S1, const LoanSet *S2) {
          assert((S1 || S2) && "unexpectedly merging 2 empty sets");
          if (!S1)
            return *S2;
          if (!S2)
            return *S1;
          return utils::join(*S1, *S2, LoanSetFactory);
        },
        // Asymmetric join is a performance win. For origins present only on one
        // branch, the loan set can be carried over as-is.
        utils::JoinKind::Asymmetric);
    return Lattice(JoinedOrigins);
  }

  /// A new loan is issued to the origin. Old loans are erased.
  Lattice transfer(Lattice In, const IssueFact &F) {
    OriginID OID = F.getOriginID();
    LoanID LID = F.getLoanID();
    return LoanPropagationLattice(OriginLoanMapFactory.add(
        In.Origins, OID,
        LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID)));
  }

  /// A flow from source to destination. If `KillDest` is true, this replaces
  /// the destination's loans with the source's. Otherwise, the source's loans
  /// are merged into the destination's.
  Lattice transfer(Lattice In, const OriginFlowFact &F) {
    OriginID DestOID = F.getDestOriginID();
    OriginID SrcOID = F.getSrcOriginID();

    LoanSet DestLoans =
        F.getKillDest() ? LoanSetFactory.getEmptySet() : getLoans(In, DestOID);
    LoanSet SrcLoans = getLoans(In, SrcOID);
    LoanSet MergedLoans = utils::join(DestLoans, SrcLoans, LoanSetFactory);

    return LoanPropagationLattice(
        OriginLoanMapFactory.add(In.Origins, DestOID, MergedLoans));
  }

  LoanSet getLoans(OriginID OID, ProgramPoint P) const {
    return getLoans(getState(P), OID);
  }

private:
  LoanSet getLoans(Lattice L, OriginID OID) const {
    if (auto *Loans = L.Origins.lookup(OID))
      return *Loans;
    return LoanSetFactory.getEmptySet();
  }
};
} // namespace internal
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LOAN_PROPAGATION_H
