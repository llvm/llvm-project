//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <future>

// enum class future_status
// {
//     ready,
//     timeout,
//     deferred
// };

#include <future>

int main()
{
    static_assert(static_cast<int>(std::future_status::ready) == 0, "");
    static_assert(static_cast<int>(std::future_status::timeout) == 1, "");
    static_assert(static_cast<int>(std::future_status::deferred) == 2, "");
}
