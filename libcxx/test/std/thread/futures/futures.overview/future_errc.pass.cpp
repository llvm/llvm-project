//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// UNSUPPORTED: c++98, c++03
// Libc++'s enum class emulation does not allow static_cast<Enum>(0) to work.

// <future>

// enum class future_errc
// {
//     broken_promise = implementation-defined,
//     future_already_retrieved = implementation-defined,
//     promise_already_satisfied = implementation-defined,
//     no_state = implementation-defined
// };

#include <future>

int main(int, char**)
{
    static_assert(std::future_errc::broken_promise != std::future_errc::future_already_retrieved, "");
    static_assert(std::future_errc::broken_promise != std::future_errc::promise_already_satisfied, "");
    static_assert(std::future_errc::broken_promise != std::future_errc::no_state, "");
    static_assert(std::future_errc::future_already_retrieved != std::future_errc::promise_already_satisfied, "");
    static_assert(std::future_errc::future_already_retrieved != std::future_errc::no_state, "");
    static_assert(std::future_errc::promise_already_satisfied != std::future_errc::no_state, "");

    static_assert(std::future_errc::broken_promise != static_cast<std::future_errc>(0), "");
    static_assert(std::future_errc::future_already_retrieved != static_cast<std::future_errc>(0), "");
    static_assert(std::future_errc::promise_already_satisfied != static_cast<std::future_errc>(0), "");
    static_assert(std::future_errc::no_state != static_cast<std::future_errc>(0), "");

  return 0;
}
