//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// On Windows Clang bugs out when both __declspec and __attribute__ are present,
// the processing goes awry preventing the definition of the types.
// XFAIL: msvc

// UNSUPPORTED: libcpp-has-no-threads
// REQUIRES: thread-safety

// <mutex>

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS

#include <mutex>

#include "test_macros.h"

std::mutex m;
int foo __attribute__((guarded_by(m)));

static void scoped() {
#if TEST_STD_VER >= 17
  std::scoped_lock<std::mutex> lock(m);
  foo++;
#endif
}

int main(int, char**) {
  scoped();
  std::lock_guard<std::mutex> lock(m);
  foo++;

  return 0;
}
