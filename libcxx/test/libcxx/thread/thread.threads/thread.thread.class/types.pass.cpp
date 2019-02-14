//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, libcpp-has-thread-api-external
// REQUIRES: libcpp-has-thread-api-pthread

// <thread>

// class thread
// {
// public:
//     typedef pthread_t native_handle_type;
//     ...
// };

#include <thread>
#include <type_traits>

int main(int, char**)
{
    static_assert((std::is_same<std::thread::native_handle_type, pthread_t>::value), "");

  return 0;
}
