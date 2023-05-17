//===-- Self contained functional header ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H

#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/macros/attributes.h"

#include <stdint.h>

namespace __llvm_libc {
namespace cpp {

/// A function type adapted from LLVM's function_ref.
/// This class does not own the callable, so it is not in general safe to
/// store a function.
template <typename Fn> class function;

template <typename Ret, typename... Params> class function<Ret(Params...)> {
  Ret (*callback)(intptr_t callable, Params... params) = nullptr;
  intptr_t callable;

  template <typename Callable>
  LIBC_INLINE static Ret callback_fn(intptr_t callable, Params... params) {
    return (*reinterpret_cast<Callable *>(callable))(
        forward<Params>(params)...);
  }

public:
  LIBC_INLINE function() = default;
  LIBC_INLINE function(decltype(nullptr)) {}
  LIBC_INLINE ~function() = default;

  template <typename Callable>
  LIBC_INLINE function(
      Callable &&callable,
      // This is not the copy-constructor.
      enable_if_t<!is_same<remove_cvref_t<Callable>, function>::value> * =
          nullptr,
      // Functor must be callable and return a suitable type.
      enable_if_t<is_void_v<Ret> ||
                  is_convertible_v<
                      decltype(declval<Callable>()(declval<Params>()...)), Ret>>
          * = nullptr)
      : callback(callback_fn<remove_reference_t<Callable>>),
        callable(reinterpret_cast<intptr_t>(&callable)) {}

  LIBC_INLINE Ret operator()(Params... params) const {
    return callback(callable, forward<Params>(params)...);
  }

  LIBC_INLINE explicit operator bool() const { return callback; }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_FUNCTIONAL_H
