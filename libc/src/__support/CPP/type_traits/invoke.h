//===-- invoke type_traits --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_H

#include "src/__support/CPP/type_traits/decay.h"
#include "src/__support/CPP/type_traits/is_base_of.h"
#include "src/__support/CPP/utility/forward.h"

// BEWARE : this implementation is not fully conformant as it doesn't take
// `cpp::reference_wrapper` into account.

namespace __llvm_libc::cpp {

namespace detail {

// Catch all function types.
template <class FunctionPtrType> struct invoke_dispatcher {
  template <class... Args>
  static auto call(FunctionPtrType &&fun, Args &&...args) {
    return cpp::forward<FunctionPtrType>(fun)(cpp::forward<Args>(args)...);
  }
};

// Catch pointer to member function types.
template <class Class, class FunctionReturnType>
struct invoke_dispatcher<FunctionReturnType Class::*> {
  using FunctionPtrType = FunctionReturnType Class::*;

  template <class T, class... Args, class DecayT = cpp::decay_t<T>>
  static auto call(FunctionPtrType fun, T &&t1, Args &&...args) {
    if constexpr (cpp::is_base_of_v<Class, DecayT>) {
      // T is a (possibly cv ref) type.
      return (cpp::forward<T>(t1).*fun)(cpp::forward<Args>(args)...);
    } else {
      // T is assumed to be a pointer type.
      return (*cpp::forward<T>(t1).*fun)(cpp::forward<Args>(args)...);
    }
  }
};

} // namespace detail

template <class Function, class... Args>
auto invoke(Function &&fun, Args &&...args) {
  return detail::invoke_dispatcher<cpp::decay_t<Function>>::call(
      cpp::forward<Function>(fun), cpp::forward<Args>(args)...);
}

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_H
