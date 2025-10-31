//===- Facts.cpp - Lifetime Analysis Facts Implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Facts.h"
#include "clang/AST/Decl.h"
#include "clang/Analysis/Analyses/PostOrderCFGView.h"

namespace clang::lifetimes::internal {

void Fact::dump(llvm::raw_ostream &OS, const LoanManager &,
                const OriginManager &) const {
  OS << "Fact (Kind: " << static_cast<int>(K) << ")\n";
}

void IssueFact::dump(llvm::raw_ostream &OS, const LoanManager &LM,
                     const OriginManager &OM) const {
  OS << "Issue (";
  LM.getLoan(getLoanID()).dump(OS);
  OS << ", ToOrigin: ";
  OM.dump(getOriginID(), OS);
  OS << ")\n";
}

void ExpireFact::dump(llvm::raw_ostream &OS, const LoanManager &LM,
                      const OriginManager &) const {
  OS << "Expire (";
  LM.getLoan(getLoanID()).dump(OS);
  OS << ")\n";
}

void OriginFlowFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                          const OriginManager &OM) const {
  OS << "OriginFlow (Dest: ";
  OM.dump(getDestOriginID(), OS);
  OS << ", Src: ";
  OM.dump(getSrcOriginID(), OS);
  OS << (getKillDest() ? "" : ", Merge");
  OS << ")\n";
}

void ReturnOfOriginFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                              const OriginManager &OM) const {
  OS << "ReturnOfOrigin (";
  OM.dump(getReturnedOriginID(), OS);
  OS << ")\n";
}

void UseFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                   const OriginManager &OM) const {
  OS << "Use (";
  OM.dump(getUsedOrigin(OM), OS);
  OS << ", " << (isWritten() ? "Write" : "Read") << ")\n";
}

void TestPointFact::dump(llvm::raw_ostream &OS, const LoanManager &,
                         const OriginManager &) const {
  OS << "TestPoint (Annotation: \"" << getAnnotation() << "\")\n";
}

llvm::StringMap<ProgramPoint> FactManager::getTestPoints() const {
  llvm::StringMap<ProgramPoint> AnnotationToPointMap;
  for (const CFGBlock *Block : BlockToFactsMap.keys()) {
    for (const Fact *F : getFacts(Block)) {
      if (const auto *TPF = F->getAs<TestPointFact>()) {
        StringRef PointName = TPF->getAnnotation();
        assert(AnnotationToPointMap.find(PointName) ==
                   AnnotationToPointMap.end() &&
               "more than one test points with the same name");
        AnnotationToPointMap[PointName] = F;
      }
    }
  }
  return AnnotationToPointMap;
}

void FactManager::dump(const CFG &Cfg, AnalysisDeclContext &AC) const {
  llvm::dbgs() << "==========================================\n";
  llvm::dbgs() << "       Lifetime Analysis Facts:\n";
  llvm::dbgs() << "==========================================\n";
  if (const Decl *D = AC.getDecl())
    if (const auto *ND = dyn_cast<NamedDecl>(D))
      llvm::dbgs() << "Function: " << ND->getQualifiedNameAsString() << "\n";
  // Print blocks in the order as they appear in code for a stable ordering.
  for (const CFGBlock *B : *AC.getAnalysis<PostOrderCFGView>()) {
    llvm::dbgs() << "  Block B" << B->getBlockID() << ":\n";
    auto It = BlockToFactsMap.find(B);
    if (It != BlockToFactsMap.end()) {
      for (const Fact *F : It->second) {
        llvm::dbgs() << "    ";
        F->dump(llvm::dbgs(), LoanMgr, OriginMgr);
      }
    }
    llvm::dbgs() << "  End of Block\n";
  }
}

} // namespace clang::lifetimes::internal
