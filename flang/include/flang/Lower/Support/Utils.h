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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

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
  return cop.value().cast<mlir::IntegerAttr>().getValue().getSExtValue();
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

#endif // FORTRAN_LOWER_SUPPORT_UTILS_H
