//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, libcpp-has-thread-api-external

// XFAIL: windows

// <condition_variable>

// class condition_variable;

// typedef pthread_cond_t* native_handle_type;
// native_handle_type native_handle();

#include <condition_variable>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((std::is_same<std::condition_variable::native_handle_type,
                                pthread_cond_t*>::value), "");
    std::condition_variable cv;
    std::condition_variable::native_handle_type h = cv.native_handle();
    assert(h != nullptr);

  return 0;
}
