//===- Facts.cpp - Lifetime Analysis Facts Implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/LifetimeSafety/LoanPropagation.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"
#include "llvm/Support/TimeProfiler.h"

namespace clang::lifetimes::internal {

void FactManager::computePersistentOrigins(const CFG &Cfg) {
  llvm::TimeTraceScope TimeProfile("ComputePersistentOrigins");

  unsigned NumOrigins = OriginMgr.getNumOrigins();
  PersistentOrigins.resize(NumOrigins);
  llvm::SmallVector<const CFGBlock *> OriginToFirstSeenBlock(NumOrigins,
                                                             nullptr);
  for (const CFGBlock *B : Cfg) {
    for (const Fact *F : getFacts(B)) {
      auto CheckOrigin = [&](OriginID OID) {
        if (PersistentOrigins.test(OID.Value))
          return;
        auto &FirstSeenBlock = OriginToFirstSeenBlock[OID.Value];
        if (FirstSeenBlock == nullptr)
          FirstSeenBlock = B;
        if (FirstSeenBlock != B) {
          // We saw this origin in more than one block.
          PersistentOrigins.set(OID.Value);
        }
      };

      switch (F->getKind()) {
      case Fact::Kind::Issue:
        CheckOrigin(F->getAs<IssueFact>()->getOriginID());
        break;
      case Fact::Kind::OriginFlow: {
        const auto *OF = F->getAs<OriginFlowFact>();
        CheckOrigin(OF->getDestOriginID());
        CheckOrigin(OF->getSrcOriginID());
        break;
      }
      case Fact::Kind::Use:
        for (const OriginList *Cur = F->getAs<UseFact>()->getUsedOrigins(); Cur;
             Cur = Cur->peelOuterOrigin())
          CheckOrigin(Cur->getOuterOriginID());
        break;
      case Fact::Kind::KillOrigin:
        CheckOrigin(F->getAs<KillOriginFact>()->getKilledOrigin());
        break;
      case Fact::Kind::OriginEscapes:
        // An escaping origin is read at the exit block but defined earlier, so
        // it spans blocks and must participate in joins.
        CheckOrigin(F->getAs<OriginEscapesFact>()->getEscapedOriginID());
        break;
      // `Expire` and `InvalidateOrigin` only ever clear an origin, so
      // misclassifying one is harmless: the clear becomes a no-op.
      case Fact::Kind::MovedOrigin:
      case Fact::Kind::Expire:
      case Fact::Kind::TestPoint:
      case Fact::Kind::InvalidateOrigin:
        break;
      }
    }
  }
}

void Fact::dump(llvm::raw_ostream &OS, const LoanManager &,
                const OriginManager &, const LoanPropagationAnalysis *) const {
  OS << "Fact (Kind: " << static_cast<int>(K) << ")\n";
}

void IssueFact::dump(llvm::raw_ostream &OS, const LoanManager &LM,
                     const OriginManager &OM,
                     const LoanPropagationAnalysis *) const {
  OS << "Issue (";
  LM.getLoan(getLoanID())->dump(OS);
  OS << ", ToOrigin: ";
  OM.dump(getOriginID(), OS);
  OS << ")\n";
}

void ExpireFact::dump(llvm::raw_ostream &OS, const LoanManager &LM,
                      const OriginManager &OM,
                      const LoanPropagationAnalysis *LPA) const {
  OS << "Expire (";
  getAccessPath().dump(OS);
  if (auto OID = getOriginID()) {
    OS << ", Origin: ";
    OM.dump(*OID, OS);
  }
  OS << ")\n";
}

void OriginFlowFact::dump(llvm::raw_ostream &OS, const LoanManager &LM,
                          const OriginManager &OM,
                          const LoanPropagationAnalysis *LPA) const {
  OS << "OriginFlow: \n";
  OS << "\tDest: ";
  OM.dump(getDestOriginID(), OS);
  if (LPA) {
    LoanSet DestinationLoans = LPA->getLoans(getDestOriginID(), this);
    if (DestinationLoans.isEmpty())
      OS << " has no loans";
    else {
      OS << " has loans to { ";
      for (LoanID LID : DestinationLoans) {
        LM.getLoan(LID)->getAccessPath().dump(OS);
        OS << " ";
      }
      OS << "}";
    }
  }
  OS << "\n";
  OS << "\tSrc:  ";
  OM.dump(getSrcOriginID(), OS);
  OS << (getKillDest() ? "" : ", Merge");
  OS << "\n";
}

void MovedOriginFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                           const OriginManager &OM,
                           const LoanPropagationAnalysis *) const {
  OS << "MovedOrigins (";
  OM.dump(getMovedOrigin(), OS);
  OS << ")\n";
}

void ReturnEscapeFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                            const OriginManager &OM,
                            const LoanPropagationAnalysis *) const {
  OS << "OriginEscapes (";
  OM.dump(getEscapedOriginID(), OS);
  OS << ", via Return)\n";
}

void FieldEscapeFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                           const OriginManager &OM,
                           const LoanPropagationAnalysis *) const {
  OS << "OriginEscapes (";
  OM.dump(getEscapedOriginID(), OS);
  OS << ", via Field)\n";
}

void GlobalEscapeFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                            const OriginManager &OM,
                            const LoanPropagationAnalysis *) const {
  OS << "OriginEscapes (";
  OM.dump(getEscapedOriginID(), OS);
  OS << ", via Global)\n";
}

void UseFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                   const OriginManager &OM,
                   const LoanPropagationAnalysis *) const {
  OS << "Use (";
  size_t NumUsedOrigins = getUsedOrigins()->getLength();
  size_t I = 0;
  for (const OriginList *Cur = getUsedOrigins(); Cur;
       Cur = Cur->peelOuterOrigin(), ++I) {
    OM.dump(Cur->getOuterOriginID(), OS);
    if (I < NumUsedOrigins - 1)
      OS << ", ";
  }
  OS << ", " << (isWritten() ? "Write" : "Read") << ")\n";
}

void InvalidateOriginFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                                const OriginManager &OM,
                                const LoanPropagationAnalysis *) const {
  OS << "InvalidateOrigin (";
  OM.dump(getInvalidatedOrigin(), OS);
  OS << ")\n";
}

void TestPointFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                         const OriginManager &,
                         const LoanPropagationAnalysis *) const {
  OS << "TestPoint (Annotation: \"" << getAnnotation() << "\")\n";
}

void KillOriginFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                          const OriginManager &OM,
                          const LoanPropagationAnalysis *) const {
  OS << "KillOrigin (";
  OM.dump(getKilledOrigin(), OS);
  OS << ")\n";
}

llvm::StringMap<ProgramPoint> FactManager::getTestPoints() const {
  llvm::StringMap<ProgramPoint> AnnotationToPointMap;
  for (const auto &BlockFacts : BlockToFacts) {
    for (const Fact *F : BlockFacts) {
      if (const auto *TPF = F->getAs<TestPointFact>()) {
        StringRef PointName = TPF->getAnnotation();
        assert(!AnnotationToPointMap.contains(PointName) &&
               "more than one test points with the same name");
        AnnotationToPointMap[PointName] = F;
      }
    }
  }
  return AnnotationToPointMap;
}

void FactManager::dump(const CFG &Cfg, AnalysisDeclContext &AC,
                       const LoanPropagationAnalysis *LPA) const {
  llvm::dbgs() << "==========================================\n";
  llvm::dbgs() << "       Lifetime Analysis Facts:\n";
  llvm::dbgs() << "==========================================\n";
  if (const Decl *D = AC.getDecl())
    if (const auto *ND = dyn_cast<NamedDecl>(D))
      llvm::dbgs() << "Function: " << ND->getQualifiedNameAsString() << "\n";
  // Print blocks in the order as they appear in code for a stable ordering.
  for (const CFGBlock *B : *AC.getAnalysis<PostOrderCFGView>()) {
    llvm::dbgs() << "  Block B" << B->getBlockID() << ":\n";
    for (const Fact *F : getFacts(B)) {
      llvm::dbgs() << "    ";
      F->dump(llvm::dbgs(), LoanMgr, OriginMgr, LPA);
    }
    llvm::dbgs() << "  End of Block\n";
  }
}

llvm::ArrayRef<const Fact *>
FactManager::getBlockContaining(ProgramPoint P) const {
  return BlockToFacts[getBlockID(P)];
}

size_t FactManager::getBlockID(ProgramPoint P) const {
  for (size_t i = 0; i < BlockToFacts.size(); ++i)
    for (const Fact *F : BlockToFacts[i])
      if (F == P)
        return i;
  llvm_unreachable("Failed to find BlockID for given ProgramPoint");
}
} // namespace clang::lifetimes::internal
