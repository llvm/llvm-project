//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <atomic>

// template <class T>
// struct atomic
// {
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
// };

// template <class T>
// struct atomic<T*>
// {
//     atomic(const atomic&) = delete;
//     atomic& operator=(const atomic&) = delete;
//     atomic& operator=(const atomic&) volatile = delete;
// };

#include <atomic>
#include <type_traits>

template <typename T>
using is_volatile_copy_assignable = std::is_assignable<volatile T&, const T&>;

int main(int, char**)
{
    static_assert(!std::is_copy_constructible<std::atomic<int> >::value, "");
    static_assert(!std::is_copy_assignable<std::atomic<int> >::value, "");
    static_assert(!is_volatile_copy_assignable<std::atomic<int> >::value, "");
    static_assert(!std::is_copy_constructible<std::atomic<int*> >::value, "");
    static_assert(!std::is_copy_assignable<std::atomic<int*> >::value, "");
    static_assert(!is_volatile_copy_assignable<std::atomic<int*> >::value, "");

    return 0;
}
