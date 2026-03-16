//===---- MCDCState.h - Per-Function MC/DC state ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Per-Function MC/DC state for PGO
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_MCDCSTATE_H
#define LLVM_CLANG_LIB_CODEGEN_MCDCSTATE_H

#include "Address.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ProfileData/Coverage/MCDCTypes.h"
#include <cassert>
#include <limits>

namespace clang {
class Stmt;
} // namespace clang

namespace clang::CodeGen::MCDC {

using namespace llvm::coverage::mcdc;

/// Per-Function MC/DC state
struct State {
  unsigned BitmapBits = 0;

  struct Decision {
    using IndicesTy = llvm::SmallVector<std::array<int, 2>>;
    static constexpr auto InvalidID = std::numeric_limits<unsigned>::max();

    unsigned BitmapIdx;
    IndicesTy Indices;
    unsigned ID = InvalidID;
    Address MCDCCondBitmapAddr = Address::invalid();

    bool isValid() const { return ID != InvalidID; }

    void update(unsigned I, IndicesTy &&X) {
      assert(isValid());
      BitmapIdx = I;
      Indices = std::move(X);
    }
  };

  llvm::DenseMap<const Stmt *, Decision> DecisionByStmt;

  struct Branch {
    ConditionID ID;
    const Stmt *DecisionStmt;
  };

  llvm::DenseMap<const Stmt *, Branch> BranchByStmt;
};

} // namespace clang::CodeGen::MCDC

#endif // LLVM_CLANG_LIB_CODEGEN_MCDCSTATE_H
