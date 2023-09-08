//===-- invoke_result type_traits -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_RESULT_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_RESULT_H

#include "src/__support/CPP/type_traits/decay.h"
#include "src/__support/CPP/type_traits/enable_if.h"
#include "src/__support/CPP/type_traits/false_type.h"
#include "src/__support/CPP/type_traits/is_base_of.h"
#include "src/__support/CPP/type_traits/is_function.h"
#include "src/__support/CPP/type_traits/true_type.h"
#include "src/__support/CPP/utility/declval.h"
#include "src/__support/CPP/utility/forward.h"

// BEWARE : this implementation is not fully conformant as it doesn't take
// `cpp::reference_wrapper` into account.

namespace __llvm_libc::cpp {

// invoke_result

namespace detail {
template <class T> struct is_reference_wrapper : cpp::false_type {};

// Disable specialization on `cpp::reference_wrapper` as it is not yet
// implemented.

// template <class U> struct
// is_reference_wrapper<cpp::reference_wrapper<U>> : cpp::true_type {};

template <class T> struct invoke_impl {
  template <class F, class... Args>
  static auto call(F &&f, Args &&...args)
      -> decltype(cpp::forward<F>(f)(cpp::forward<Args>(args)...));
};

template <class B, class MT> struct invoke_impl<MT B::*> {
  template <class T, class Td = cpp::decay_t<T>,
            class = cpp::enable_if_t<cpp::is_base_of_v<B, Td>>>
  static auto get(T &&t) -> T &&;

  template <class T, class Td = cpp::decay_t<T>,
            class = cpp::enable_if_t<is_reference_wrapper<Td>::value>>
  static auto get(T &&t) -> decltype(t.get());

  template <class T, class Td = cpp::decay_t<T>,
            class = cpp::enable_if_t<!cpp::is_base_of_v<B, Td>>,
            class = cpp::enable_if_t<!is_reference_wrapper<Td>::value>>
  static auto get(T &&t) -> decltype(*cpp::forward<T>(t));

  template <class T, class... Args, class MT1,
            class = cpp::enable_if_t<cpp::is_function_v<MT1>>>
  static auto call(MT1 B::*pmf, T &&t, Args &&...args)
      -> decltype((invoke_impl::get(cpp::forward<T>(t)).*
                   pmf)(cpp::forward<Args>(args)...));

  template <class T>
  static auto call(MT B::*pmd, T &&t)
      -> decltype(invoke_impl::get(cpp::forward<T>(t)).*pmd);
};

template <class F, class... Args, class Fd = typename cpp::decay_t<F>>
auto INVOKE(F &&f, Args &&...args)
    -> decltype(invoke_impl<Fd>::call(cpp::forward<F>(f),
                                      cpp::forward<Args>(args)...));

template <typename AlwaysVoid, typename, typename...> struct invoke_result {};
template <typename F, typename... Args>
struct invoke_result<decltype(void(detail::INVOKE(cpp::declval<F>(),
                                                  cpp::declval<Args>()...))),
                     F, Args...> {
  using type =
      decltype(detail::INVOKE(cpp::declval<F>(), cpp::declval<Args>()...));
};
} // namespace detail

template <class F, class... ArgTypes>
struct invoke_result : detail::invoke_result<void, F, ArgTypes...> {};

template <class F, class... ArgTypes>
using invoke_result_t = typename invoke_result<F, ArgTypes...>::type;

} // namespace __llvm_libc::cpp

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_TYPE_TRAITS_INVOKE_RESULT_H
