//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__assert>
#include <__config>
#include <cstdio>
#include <cstdlib>

_LIBCPP_BEGIN_NAMESPACE_STD

_LIBCPP_WEAK
void __libcpp_assertion_handler(char const* __file, int __line, char const* __expression, char const* __message) {
  std::fprintf(stderr, "%s:%d: libc++ assertion '%s' failed. %s\n", __file, __line, __expression, __message);
  std::abort();
}

_LIBCPP_END_NAMESPACE_STD
