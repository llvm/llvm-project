//===-- Lower/Utils.h -- utilities ------------------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_UTILS_H
#define FORTRAN_LOWER_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Parser/char-block.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

namespace Fortran {
namespace semantics {
class Symbol;
}

namespace evaluate {
template <typename A>
class Expr;
struct SomeType;
} // namespace evaluate

namespace common {
template <typename A>
class Reference;
}

namespace lower::pft {
struct Evaluation;

using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
using SymbolRef = Fortran::common::Reference<const Fortran::semantics::Symbol>;
using Label = std::uint64_t;
using LabelSet = llvm::SmallSet<Label, 4>;
using SymbolLabelMap = llvm::DenseMap<SymbolRef, LabelSet>;
using LabelEvalMap = llvm::DenseMap<Label, Evaluation *>;
} // namespace lower::pft
} // namespace Fortran

/// Convert an F18 CharBlock to an LLVM StringRef
inline llvm::StringRef toStringRef(const Fortran::parser::CharBlock &cb) {
  return {cb.begin(), cb.size()};
}

/// Template helper to remove Fortran::common::Indirection wrappers.
template <typename A>
const A &removeIndirection(const A &a) {
  return a;
}
template <typename A>
const A &removeIndirection(const Fortran::common::Indirection<A> &a) {
  return a.value();
}

#endif // FORTRAN_LOWER_UTILS_H
