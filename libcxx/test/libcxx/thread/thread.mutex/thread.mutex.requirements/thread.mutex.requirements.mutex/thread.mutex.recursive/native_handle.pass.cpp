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

// <mutex>

// class recursive_mutex;

// typedef pthread_mutex_t* native_handle_type;
// native_handle_type native_handle();

#include <mutex>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::recursive_mutex m;
    pthread_mutex_t* h = m.native_handle();
    assert(h);

  return 0;
}
