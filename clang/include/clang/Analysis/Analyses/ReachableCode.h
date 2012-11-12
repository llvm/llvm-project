//===- ReachableCode.h -----------------------------------------*- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A flow-sensitive, path-insensitive analysis of unreachable code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_REACHABLECODE_H
#define LLVM_CLANG_REACHABLECODE_H

#include "clang/Basic/SourceLocation.h"

//===----------------------------------------------------------------------===//
// Forward declarations.
//===----------------------------------------------------------------------===//

namespace llvm {
  class BitVector;
}

namespace clang {
  class AnalysisDeclContext;
  class CFGBlock;
}

//===----------------------------------------------------------------------===//
// API.
//===----------------------------------------------------------------------===//

namespace clang {
namespace reachable_code {

class Callback {
  virtual void anchor();
public:
  virtual ~Callback() {}
  virtual void HandleUnreachable(SourceLocation L, SourceRange R1,
                                 SourceRange R2) = 0;
};

/// ScanReachableFromBlock - Mark all blocks reachable from Start.
/// Returns the total number of blocks that were marked reachable.  
unsigned ScanReachableFromBlock(const CFGBlock *Start,
                                llvm::BitVector &Reachable);

void FindUnreachableCode(AnalysisDeclContext &AC, Callback &CB);

}} // end namespace clang::reachable_code

#endif
