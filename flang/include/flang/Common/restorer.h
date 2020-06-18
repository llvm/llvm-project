//===-- include/flang/Common/restorer.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Utility: before overwriting a variable, capture its value and
// ensure that it will be restored when the Restorer goes out of scope.
//
// int x{3};
// {
//   auto save{common::ScopedSet(x, 4)};
//   // x is now 4
// }
// // x is back to 3

#ifndef FORTRAN_COMMON_RESTORER_H_
#define FORTRAN_COMMON_RESTORER_H_
#include "idioms.h"
namespace Fortran::common {
template <typename A> class Restorer {
public:
  explicit Restorer(A &p) : p_{p}, original_{std::move(p)} {}
  ~Restorer() { p_ = std::move(original_); }

private:
  A &p_;
  A original_;
};

template <typename A, typename B>
common::IfNoLvalue<Restorer<A>, B> ScopedSet(A &to, B &&from) {
  Restorer<A> result{to};
  to = std::move(from);
  return result;
}
template <typename A, typename B>
common::IfNoLvalue<Restorer<A>, B> ScopedSet(A &to, const B &from) {
  Restorer<A> result{to};
  to = from;
  return result;
}
} // namespace Fortran::common
#endif // FORTRAN_COMMON_RESTORER_H_
