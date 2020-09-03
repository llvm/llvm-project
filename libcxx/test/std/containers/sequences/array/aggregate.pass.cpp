//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure std::array is an aggregate type.
// We can only check this in C++17 and above, because we don't have the
// trait before that.
// UNSUPPORTED: c++03, c++11, c++14

// libc++ doesn't implement std::is_aggregate on GCC 5 and GCC 6.
// UNSUPPORTED: libc++ && gcc-5
// UNSUPPORTED: libc++ && gcc-6

#include <array>
#include <type_traits>

template <typename T>
void check_aggregate()
{
    static_assert(std::is_aggregate<std::array<T, 0> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 1> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 2> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 3> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 4> >::value, "");
}

struct Empty { };
struct Trivial { int i; int j; };
struct NonTrivial {
    int i; int j;
    NonTrivial(NonTrivial const&) { }
};

int main(int, char**)
{
    check_aggregate<char>();
    check_aggregate<int>();
    check_aggregate<long>();
    check_aggregate<float>();
    check_aggregate<double>();
    check_aggregate<long double>();
    check_aggregate<Empty>();
    check_aggregate<Trivial>();
    check_aggregate<NonTrivial>();

    return 0;
}
