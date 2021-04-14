//===-- MaskExpr.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_MASKEXPR_H
#define FORTRAN_LOWER_MASKEXPR_H

#include "StatementContext.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace Fortran::evaluate {
class SomeType;
template <typename>
class Expr;
} // namespace Fortran::evaluate

namespace Fortran::lower {

/// Collect WHERE construct mask expressions in the bridge and forward lowering
/// results in an "evaluate once" semantics. See 10.2.3.2p3, 10.2.3.2p13, etc.
class MaskExpr {
public:
  using FrontEndMaskExpr =
      const Fortran::evaluate::Expr<Fortran::evaluate::SomeType> *;

  bool empty() const { return masks.empty(); }

  void growStack() {
    if (empty())
      stmtCtx.reset();
    masks.push_back(llvm::SmallVector<FrontEndMaskExpr>{});
  }

  void shrinkStack() {
    assert(!empty());
    masks.pop_back();
    if (empty())
      stmtCtx.finalize();
  }

  void append(FrontEndMaskExpr e) {
    assert(!empty());
    masks.back().push_back(e);
  }

  llvm::SmallVector<FrontEndMaskExpr> getExprs() const {
    auto maskList = masks[0];
    for (unsigned i = 1, d = masks.size(); i < d; ++i)
      maskList.append(masks[i].begin(), masks[i].end());
    return maskList;
  }

  // Map each mask expression back to the temporary holding the initial
  // evaluation results.
  llvm::DenseMap<FrontEndMaskExpr, mlir::Value> vmap;

  // Inflate the statement context for the entire WHERE construct. We have to
  // cache the mask expression results across the entire construct.
  Fortran::lower::StatementContext stmtCtx;

private:
  // Stack of WHERE constructs, each building a list of mask expressions.
  llvm::SmallVector<llvm::SmallVector<FrontEndMaskExpr>> masks;
};

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_MASKEXPR_H
