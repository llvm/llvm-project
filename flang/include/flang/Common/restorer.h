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
  explicit Restorer(A &p, A original) : p_{p}, original_{std::move(original)} {}
  ~Restorer() { p_ = std::move(original_); }

  // Inhibit any recreation of this restorer that would result in two restorers
  // trying to restore the same reference.
  Restorer(const Restorer &) = delete;
  Restorer(Restorer &&that) = delete;
  const Restorer &operator=(const Restorer &) = delete;
  const Restorer &operator=(Restorer &&that) = delete;

private:
  A &p_;
  A original_;
};

template <typename A, typename B>
common::IfNoLvalue<Restorer<A>, B> ScopedSet(A &to, B &&from) {
  A original{std::move(to)};
  to = std::move(from);
  return Restorer<A>{to, std::move(original)};
}
template <typename A, typename B>
common::IfNoLvalue<Restorer<A>, B> ScopedSet(A &to, const B &from) {
  A original{std::move(to)};
  to = from;
  return Restorer<A>{to, std::move(original)};
}
} // namespace Fortran::common
#endif // FORTRAN_COMMON_RESTORER_H_
