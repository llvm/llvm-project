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
    static_assert(std::launch::any == (std::launch::async | std::launch::deferred), "");
    static_assert(std::launch::async == 1, "");
    static_assert(std::launch::deferred == 2, "");
}
