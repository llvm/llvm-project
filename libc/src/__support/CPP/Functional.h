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

template <typename Func> class Function;

template <typename Ret, typename... Params> class Function<Ret(Params...)> {
  Ret (*func)(Params...) = nullptr;

public:
  constexpr Function() = default;
  template <typename Func> constexpr Function(Func &&f) : func(f) {}

  constexpr Ret operator()(Params... params) { return func(params...); }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H
