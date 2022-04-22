//===-- Lower/Support/Utils.h -- utilities ----------------------*- C++ -*-===//
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

#ifndef FORTRAN_LOWER_SUPPORT_UTILS_H
#define FORTRAN_LOWER_SUPPORT_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/tools.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace Fortran::lower {
using SomeExpr = Fortran::evaluate::Expr<Fortran::evaluate::SomeType>;
}

//===----------------------------------------------------------------------===//
// Small inline helper functions to deal with repetitive, clumsy conversions.
//===----------------------------------------------------------------------===//

/// Convert an F18 CharBlock to an LLVM StringRef.
inline llvm::StringRef toStringRef(const Fortran::parser::CharBlock &cb) {
  return {cb.begin(), cb.size()};
}

namespace fir {
/// Return the integer value of a arith::ConstantOp.
inline std::int64_t toInt(mlir::arith::ConstantOp cop) {
  return cop.getValue().cast<mlir::IntegerAttr>().getValue().getSExtValue();
}
} // namespace fir

/// Template helper to remove Fortran::common::Indirection wrappers.
template <typename A>
const A &removeIndirection(const A &a) {
  return a;
}
template <typename A>
const A &removeIndirection(const Fortran::common::Indirection<A> &a) {
  return a.value();
}

/// Clone subexpression and wrap it as a generic `Fortran::evaluate::Expr`.
template <typename A>
static Fortran::lower::SomeExpr toEvExpr(const A &x) {
  return Fortran::evaluate::AsGenericExpr(Fortran::common::Clone(x));
}

template <Fortran::common::TypeCategory FROM>
static Fortran::lower::SomeExpr ignoreEvConvert(
    const Fortran::evaluate::Convert<
        Fortran::evaluate::Type<Fortran::common::TypeCategory::Integer, 8>,
        FROM> &x) {
  return toEvExpr(x.left());
}
template <typename A>
static Fortran::lower::SomeExpr ignoreEvConvert(const A &x) {
  return toEvExpr(x);
}

/// A vector subscript expression may be wrapped with a cast to INTEGER*8.
/// Get rid of it here so the vector can be loaded. Add it back when
/// generating the elemental evaluation (inside the loop nest).
inline Fortran::lower::SomeExpr
ignoreEvConvert(const Fortran::evaluate::Expr<Fortran::evaluate::Type<
                    Fortran::common::TypeCategory::Integer, 8>> &x) {
  return std::visit([](const auto &v) { return ignoreEvConvert(v); }, x.u);
}

/// Zip two containers of the same size together and flatten the pairs. `flatZip
/// [1;2] [3;4]` yields `[1;3;2;4]`.
template <typename A>
A flatZip(const A &container1, const A &container2) {
  assert(container1.size() == container2.size());
  A result;
  for (auto [e1, e2] : llvm::zip(container1, container2)) {
    result.emplace_back(e1);
    result.emplace_back(e2);
  }
  return result;
}

#endif // FORTRAN_LOWER_SUPPORT_UTILS_H
