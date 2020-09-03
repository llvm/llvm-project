//===-- Lower/Utils.h -- utilities ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_UTILS_H
#define FORTRAN_LOWER_UTILS_H

#include "flang/Common/indirection.h"
#include "flang/Parser/char-block.h"
#include "llvm/ADT/StringRef.h"

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
