//===-- Implementation of strcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strcmp.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_strcmp.h"

#ifndef LIBC_COPT_STRCMP_IMPL
#define LIBC_COPT_STRCMP_IMPL element
#endif

namespace LIBC_NAMESPACE_DECL {
namespace internal {

namespace element {
LIBC_INLINE int strcmp(const char *left, const char *right) {
  auto comp = [](char l, char r) -> int {
    return static_cast<unsigned char>(l) - static_cast<unsigned char>(r);
  };
  return inline_strcmp(left, right, comp);
}
} // namespace element

static constexpr auto &strcmp_impl = LIBC_COPT_STRCMP_IMPL::strcmp;

} // namespace internal

LLVM_LIBC_FUNCTION(int, strcmp, (const char *left, const char *right)) {
  return internal::strcmp_impl(left, right);
}

} // namespace LIBC_NAMESPACE_DECL
