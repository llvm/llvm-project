//===-- Self contained functional header ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H

namespace __llvm_libc {
namespace cpp {

template <typename F> class function;

template <typename R, typename... Args> class function<R(Args...)> {
  R (*func)(Args...) = nullptr;

public:
  constexpr function() = default;
  template <typename F> constexpr function(F &&f) : func(f) {}

  constexpr R operator()(Args... params) { return func(params...); }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H
