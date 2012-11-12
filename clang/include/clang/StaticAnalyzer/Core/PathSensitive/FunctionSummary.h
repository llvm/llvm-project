//== FunctionSummary.h - Stores summaries of functions. ------------*- C++ -*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a summary of a function gathered/used by static analyzes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_FUNCTIONSUMMARY_H
#define LLVM_CLANG_GR_FUNCTIONSUMMARY_H

#include <deque>
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/BitVector.h"

namespace clang {
namespace ento {
typedef std::deque<Decl*> SetOfDecls;
typedef llvm::DenseSet<const Decl*> SetOfConstDecls;

class FunctionSummariesTy {
  struct FunctionSummary {
    /// True if this function has reached a max block count while inlined from
    /// at least one call site.
    bool MayReachMaxBlockCount;

    /// Total number of blocks in the function.
    unsigned TotalBasicBlocks;

    /// Marks the IDs of the basic blocks visited during the analyzes.
    llvm::BitVector VisitedBasicBlocks;

    FunctionSummary() :
      MayReachMaxBlockCount(false),
      TotalBasicBlocks(0),
      VisitedBasicBlocks(0) {}
  };

  typedef llvm::DenseMap<const Decl*, FunctionSummary*> MapTy;
  MapTy Map;

public:
  ~FunctionSummariesTy();

  MapTy::iterator findOrInsertSummary(const Decl *D) {
    MapTy::iterator I = Map.find(D);
    if (I != Map.end())
      return I;
    FunctionSummary *DS = new FunctionSummary();
    I = Map.insert(std::pair<const Decl*, FunctionSummary*>(D, DS)).first;
    assert(I != Map.end());
    return I;
  }

  void markReachedMaxBlockCount(const Decl* D) {
    MapTy::iterator I = findOrInsertSummary(D);
    I->second->MayReachMaxBlockCount = true;
  }

  bool hasReachedMaxBlockCount(const Decl* D) {
  MapTy::const_iterator I = Map.find(D);
    if (I != Map.end())
      return I->second->MayReachMaxBlockCount;
    return false;
  }

  void markVisitedBasicBlock(unsigned ID, const Decl* D, unsigned TotalIDs) {
    MapTy::iterator I = findOrInsertSummary(D);
    llvm::BitVector &Blocks = I->second->VisitedBasicBlocks;
    assert(ID < TotalIDs);
    if (TotalIDs > Blocks.size()) {
      Blocks.resize(TotalIDs);
      I->second->TotalBasicBlocks = TotalIDs;
    }
    Blocks[ID] = true;
  }

  unsigned getNumVisitedBasicBlocks(const Decl* D) {
    MapTy::const_iterator I = Map.find(D);
      if (I != Map.end())
        return I->second->VisitedBasicBlocks.count();
    return 0;
  }

  /// Get the percentage of the reachable blocks.
  unsigned getPercentBlocksReachable(const Decl *D) {
    MapTy::const_iterator I = Map.find(D);
      if (I != Map.end())
        return ((I->second->VisitedBasicBlocks.count() * 100) /
                 I->second->TotalBasicBlocks);
    return 0;
  }

  unsigned getTotalNumBasicBlocks();
  unsigned getTotalNumVisitedBasicBlocks();

};

}} // end clang ento namespaces

#endif
