//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// enum class future_errc
// {
//     broken_promise,
//     future_already_retrieved,
//     promise_already_satisfied,
//     no_state
// };

#include <future>

int main()
{
    static_assert(static_cast<int>(std::future_errc::broken_promise) == 0, "");
    static_assert(static_cast<int>(std::future_errc::future_already_retrieved) == 1, "");
    static_assert(static_cast<int>(std::future_errc::promise_already_satisfied) == 2, "");
    static_assert(static_cast<int>(std::future_errc::no_state) == 3, "");
}
