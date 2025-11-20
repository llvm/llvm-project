//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

//
// Check that libc++ honors when __SANITIZER_DISABLE_CONTAINER_OVERFLOW__ is set.
//

// RUN: %{cxx} %s %{flags} %{compile_flags} -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__
// RUN: %{cxx} %s %{flags} %{compile_flags} -fsanitize=address -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__

#include <vector>
#include <string>
#include <deque>

#if _LIBCPP_ENABLE_ASAN_CONTAINER_CHECKS
#  error "Container overflow checks should be disabled in libc++"
#endif

int main(int, char**) {}
