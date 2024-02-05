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
#include "src/__support/macros/properties/architectures.h"

#ifndef LLVM_LIBC_FUNCTION_ATTR
#define LLVM_LIBC_FUNCTION_ATTR
#endif

// MacOS needs to be excluded because it does not support aliasing.
#if defined(LIBC_COPT_PUBLIC_PACKAGING) && (!defined(__APPLE__))
#define LLVM_LIBC_FUNCTION_IMPL(type, name, arglist)                           \
  LLVM_LIBC_FUNCTION_ATTR decltype(LIBC_NAMESPACE::name)                       \
      __##name##_impl__ __asm__(#name);                                        \
  decltype(LIBC_NAMESPACE::name) name [[gnu::alias(#name)]];                   \
  type __##name##_impl__ arglist
#else
#define LLVM_LIBC_FUNCTION_IMPL(type, name, arglist) type name arglist
#endif

// This extra layer of macro allows `name` to be a macro to rename a function.
#define LLVM_LIBC_FUNCTION(type, name, arglist)                                \
  LLVM_LIBC_FUNCTION_IMPL(type, name, arglist)

namespace LIBC_NAMESPACE {
namespace internal {
LIBC_INLINE constexpr bool same_string(char const *lhs, char const *rhs) {
  for (; *lhs || *rhs; ++lhs, ++rhs)
    if (*lhs != *rhs)
      return false;
  return true;
}
} // namespace internal
} // namespace LIBC_NAMESPACE

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
