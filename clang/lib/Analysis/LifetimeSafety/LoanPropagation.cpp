// TODO: Add file header
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"

namespace clang::lifetimes::internal {

void LoanPropagationLattice::dump(llvm::raw_ostream &OS) const {
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

using Lattice = LoanPropagationAnalysis::Lattice;

/// Merges two lattices by taking the union of loans for each origin.
// TODO(opt): Keep the state small by removing origins which become dead.
Lattice LoanPropagationAnalysis::join(Lattice A, Lattice B) {
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
Lattice LoanPropagationAnalysis::transfer(Lattice In, const IssueFact &F) {
  OriginID OID = F.getOriginID();
  LoanID LID = F.getLoanID();
  return LoanPropagationLattice(OriginLoanMapFactory.add(
      In.Origins, OID, LoanSetFactory.add(LoanSetFactory.getEmptySet(), LID)));
}

/// A flow from source to destination. If `KillDest` is true, this replaces
/// the destination's loans with the source's. Otherwise, the source's loans
/// are merged into the destination's.
Lattice LoanPropagationAnalysis::transfer(Lattice In, const OriginFlowFact &F) {
  OriginID DestOID = F.getDestOriginID();
  OriginID SrcOID = F.getSrcOriginID();

  LoanSet DestLoans =
      F.getKillDest() ? LoanSetFactory.getEmptySet() : getLoans(In, DestOID);
  LoanSet SrcLoans = getLoans(In, SrcOID);
  LoanSet MergedLoans = utils::join(DestLoans, SrcLoans, LoanSetFactory);

  return LoanPropagationLattice(
      OriginLoanMapFactory.add(In.Origins, DestOID, MergedLoans));
}
} // namespace clang::lifetimes::internal
