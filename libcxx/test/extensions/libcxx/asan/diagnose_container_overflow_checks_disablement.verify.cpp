//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// Check that we diagnose when libc++ has been built with ASAN instrumentation
// and the user requests turning off the ASAN container checks. Since that is
// impossible to implement, we diagnose this with an error instead.
//
// REQUIRES: libcpp-instrumented-with-asan
// ADDITIONAL_COMPILE_FLAGS: -D__SANITIZER_DISABLE_CONTAINER_OVERFLOW__

#include <deque>
#include <string>
#include <vector>

// expected-error@*:* {{We can't disable ASAN container checks when libc++ has been built with ASAN container checks enabled}}
