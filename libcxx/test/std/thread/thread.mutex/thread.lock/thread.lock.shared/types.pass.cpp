//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03, c++11


// <shared_mutex>

// template <class Mutex>
// class shared_lock
// {
// public:
//     typedef Mutex mutex_type;
//     ...
// };

#include <shared_mutex>
#include <type_traits>

int main()
{
    static_assert((std::is_same<std::shared_lock<std::mutex>::mutex_type,
                   std::mutex>::value), "");
}
