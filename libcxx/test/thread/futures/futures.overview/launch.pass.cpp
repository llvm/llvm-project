//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// enum class launch
// {
//     async = 1,
//     deferred = 2,
//     any = async | deferred
// };

#include <future>

int main()
{
    static_assert(static_cast<int>(std::launch::any) ==
                 (static_cast<int>(std::launch::async) | static_cast<int>(std::launch::deferred)), "");
    static_assert(static_cast<int>(std::launch::async) == 1, "");
    static_assert(static_cast<int>(std::launch::deferred) == 2, "");
}
