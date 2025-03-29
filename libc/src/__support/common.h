//===-- Common internal contructs -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_COMMON_H
#define LLVM_LIBC_SRC___SUPPORT_COMMON_H

#ifndef LIBC_NAMESPACE
#error "LIBC_NAMESPACE macro is not defined."
#endif

#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

#ifndef LLVM_LIBC_FUNCTION_ATTR
#define LLVM_LIBC_FUNCTION_ATTR
#endif

// clang-format off
// Allow each function `func` to have extra attributes specified by defining:
// `LLVM_LIBC_FUNCTION_ATTR_func` macro, which should always start with
// "LLVM_LIBC_EMPTY, "
//
// For examples:
// #define LLVM_LIBC_FUNCTION_ATTR_memcpy LLVM_LIBC_EMPTY, [[gnu::weak]]
// #define LLVM_LIBC_FUNCTION_ATTR_memchr LLVM_LIBC_EMPTY, [[gnu::weak]] [[gnu::visibility("default")]]
// clang-format on
#define LLVM_LIBC_EMPTY

#define GET_SECOND(first, second, ...) second
#define EXPAND_THEN_SECOND(name) GET_SECOND(name, LLVM_LIBC_EMPTY)

#define LLVM_LIBC_ATTR(name) EXPAND_THEN_SECOND(LLVM_LIBC_FUNCTION_ATTR_##name)

// MacOS needs to be excluded because it does not support [[gnu::aliasing]].
#ifndef __APPLE__

#if defined(LIBC_COPT_PUBLIC_PACKAGING)
#define LLVM_LIBC_FUNCTION(type, name, arglist)                                \
  LLVM_LIBC_ATTR(name)                                                         \
  LLVM_LIBC_FUNCTION_ATTR decltype(LIBC_NAMESPACE::name)                       \
      __##name##_impl__ asm(#name);                                            \
  decltype(LIBC_NAMESPACE::name) name [[gnu::alias(#name)]];                   \
  type __##name##_impl__ arglist

#define LLVM_LIBC_ALIAS(name, func)                                            \
  decltype(LIBC_NAMESPACE::name) LIBC_NAMESPACE::name [[gnu::alias(#func)]];   \
  extern "C" decltype(LIBC_NAMESPACE::name) name [[gnu::alias(#func)]];        \
  static_assert(true, "Require semicolon")
#else
#define LLVM_LIBC_FUNCTION(type, name, arglist)                                \
  LLVM_LIBC_ATTR(name)                                                         \
  LLVM_LIBC_FUNCTION_ATTR decltype(LIBC_NAMESPACE::name)                       \
      __##name##_impl__ asm("__" #name "_impl__");                             \
  decltype(LIBC_NAMESPACE::name) name [[gnu::alias("__" #name "_impl__")]];    \
  type __##name##_impl__ arglist

#define LLVM_LIBC_ALIAS(name, func)                                            \
  decltype(LIBC_NAMESPACE::name) LIBC_NAMESPACE::name                          \
      [[gnu::alias("__" #func "_impl__")]];                                    \
  static_assert(true, "Require semicolon")
#endif // LIBC_COPT_PUBLIC_PACKAGING

#else

#define LLVM_LIBC_FUNCTION(type, name, arglist) type name arglist

#define LLVM_LIBC_ALIAS(name, func) static_assert(true, "Require semicolon")

#endif // !__APPLE__

namespace LIBC_NAMESPACE_DECL {
namespace internal {
LIBC_INLINE constexpr bool same_string(char const *lhs, char const *rhs) {
  for (; *lhs || *rhs; ++lhs, ++rhs)
    if (*lhs != *rhs)
      return false;
  return true;
}
} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

#define __LIBC_MACRO_TO_STRING(str) #str
#define LIBC_MACRO_TO_STRING(str) __LIBC_MACRO_TO_STRING(str)

// LLVM_LIBC_IS_DEFINED checks whether a particular macro is defined.
// Usage: constexpr bool kUseAvx = LLVM_LIBC_IS_DEFINED(__AVX__);
//
// This works by comparing the stringified version of the macro with and without
// evaluation. If FOO is not undefined both stringifications yield "FOO". If FOO
// is defined, one stringification yields "FOO" while the other yields its
// stringified value "1".
#define LLVM_LIBC_IS_DEFINED(macro)                                            \
  !LIBC_NAMESPACE::internal::same_string(                                      \
      LLVM_LIBC_IS_DEFINED__EVAL_AND_STRINGIZE(macro), #macro)
#define LLVM_LIBC_IS_DEFINED__EVAL_AND_STRINGIZE(s) #s

#endif // LLVM_LIBC_SRC___SUPPORT_COMMON_H
