//===-- Lower/PFTDefs.h -- shared PFT info ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_PFTDEFS_H
#define FORTRAN_LOWER_PFTDEFS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Block;
}

namespace Fortran {
namespace semantics {
class Symbol;
class SemanticsContext;
class Scope;
} // namespace semantics

namespace evaluate {
template <typename A>
class Expr;
struct SomeType;
} // namespace evaluate

namespace common {
template <typename A>
class Reference;
}

namespace lower {

bool definedInCommonBlock(const semantics::Symbol &sym);
bool defaultRecursiveFunctionSetting();

namespace pft {

struct Evaluation;

using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
using SymbolRef = Fortran::common::Reference<const Fortran::semantics::Symbol>;
using Label = std::uint64_t;
using LabelSet = llvm::SmallSet<Label, 4>;
using SymbolLabelMap = llvm::DenseMap<SymbolRef, LabelSet>;
using LabelEvalMap = llvm::DenseMap<Label, Evaluation *>;

} // namespace pft
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_PFTDEFS_H
