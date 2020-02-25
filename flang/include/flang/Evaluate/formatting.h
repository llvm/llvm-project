//===-- include/flang/Evaluate/formatting.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_FORMATTING_H_
#define FORTRAN_EVALUATE_FORMATTING_H_

// It is inconvenient in C++ to have std::ostream::operator<<() as a direct
// friend function of a class template with many instantiations, so the
// various representational class templates in lib/Evaluate format themselves
// via AsFortran(std::ostream &) member functions, which the operator<<()
// overload below will call.  Others have AsFortran() member functions that
// return strings.
//
// This header is meant to be included by the headers that define the several
// representational class templates that need it, not by external clients.

#include "flang/Common/indirection.h"
#include <optional>
#include <ostream>
#include <type_traits>

namespace Fortran::evaluate {

extern bool formatForPGF90;

template<typename A>
auto operator<<(std::ostream &o, const A &x) -> decltype(x.AsFortran(o)) {
  return x.AsFortran(o);
}

template<typename A>
auto operator<<(std::ostream &o, const A &x) -> decltype(o << x.AsFortran()) {
  return o << x.AsFortran();
}

template<typename A, bool COPYABLE>
auto operator<<(
    std::ostream &o, const Fortran::common::Indirection<A, COPYABLE> &x)
    -> decltype(o << x.value()) {
  return o << x.value();
}

template<typename A>
auto operator<<(std::ostream &o, const std::optional<A> &x)
    -> decltype(o << *x) {
  if (x) {
    o << *x;
  } else {
    o << "(nullopt)";
  }
  return o;
}
}
#endif  // FORTRAN_EVALUATE_FORMATTING_H_
