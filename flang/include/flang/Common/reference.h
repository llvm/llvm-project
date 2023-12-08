//===-- include/flang/Common/reference.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements a better std::reference_wrapper<> template class with
// move semantics, equality testing, and member access.
// Use Reference<A> in place of a real A& reference when assignability is
// required; safer than a bare pointer because it's guaranteed to not be null.

#ifndef FORTRAN_COMMON_REFERENCE_H_
#define FORTRAN_COMMON_REFERENCE_H_
#include <type_traits>
namespace Fortran::common {
template <typename A> class Reference {
public:
  using type = A;
  Reference(type &x) : p_{&x} {}
  Reference(const Reference &that) : p_{that.p_} {}
  Reference(Reference &&that) : p_{that.p_} {}
  Reference &operator=(const Reference &that) {
    p_ = that.p_;
    return *this;
  }
  Reference &operator=(Reference &&that) {
    p_ = that.p_;
    return *this;
  }

  // Implicit conversions to references are supported only for
  // const-qualified types in order to avoid any pernicious
  // creation of a temporary copy in cases like:
  //   Reference<type> ref;
  //   const Type &x{ref};  // creates ref to temp copy!
  operator std::conditional_t<std::is_const_v<type>, type &, void>()
      const noexcept {
    if constexpr (std::is_const_v<type>) {
      return *p_;
    }
  }

  type &get() const noexcept { return *p_; }
  type *operator->() const { return p_; }
  type &operator*() const { return *p_; }

  bool operator==(std::add_const_t<A> &that) const {
    return p_ == &that || *p_ == that;
  }
  bool operator!=(std::add_const_t<A> &that) const { return !(*this == that); }
  bool operator==(const Reference &that) const {
    return p_ == that.p_ || *this == *that.p_;
  }
  bool operator!=(const Reference &that) const { return !(*this == that); }

private:
  type *p_; // never null
};
template <typename A> Reference(A &) -> Reference<A>;
} // namespace Fortran::common
#endif
