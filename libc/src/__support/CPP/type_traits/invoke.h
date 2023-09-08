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
#include "src/__support/CPP/type_traits/invoke_result.h"
#include "src/__support/CPP/type_traits/is_base_of.h"
#include "src/__support/CPP/type_traits/is_function.h"
#include "src/__support/CPP/type_traits/is_member_pointer.h"
#include "src/__support/CPP/type_traits/is_object.h"
#include "src/__support/CPP/utility/forward.h"

// BEWARE : this implementation is not fully conformant as it doesn't take
// `cpp::reference_wrapper` into account. It also bypasses noexcept detection.

namespace __llvm_libc::cpp {

// invoke
namespace detail {
template <class> constexpr bool is_reference_wrapper_v = false;

// Disable specialization on `cpp::reference_wrapper` as it is not yet
// implemented.

// template <class U>
// constexpr bool is_reference_wrapper_v<cpp::reference_wrapper<U>> = true;

template <class C, class Pointed, class T1, class... Args>
constexpr decltype(auto) invoke_memptr(Pointed C::*f, T1 &&t1, Args &&...args) {
  if constexpr (cpp::is_function_v<Pointed>) {
    if constexpr (cpp::is_base_of_v<C, cpp::decay_t<T1>>)
      return (cpp::forward<T1>(t1).*f)(cpp::forward<Args>(args)...);
    else if constexpr (is_reference_wrapper_v<cpp::decay_t<T1>>)
      return (t1.get().*f)(cpp::forward<Args>(args)...);
    else
      return ((*cpp::forward<T1>(t1)).*f)(cpp::forward<Args>(args)...);
  } else {
    static_assert(cpp::is_object_v<Pointed> && sizeof...(args) == 0);
    if constexpr (cpp::is_base_of_v<C, cpp::decay_t<T1>>)
      return cpp::forward<T1>(t1).*f;
    else if constexpr (is_reference_wrapper_v<cpp::decay_t<T1>>)
      return t1.get().*f;
    else
      return (*cpp::forward<T1>(t1)).*f;
  }
}
} // namespace detail

template <class F, class... Args>
constexpr cpp::invoke_result_t<F, Args...> invoke(F &&f, Args &&...args) {
  if constexpr (cpp::is_member_pointer_v<cpp::decay_t<F>>)
    return detail::invoke_memptr(f, cpp::forward<Args>(args)...);
  else
    return cpp::forward<F>(f)(cpp::forward<Args>(args)...);
}
} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_H
