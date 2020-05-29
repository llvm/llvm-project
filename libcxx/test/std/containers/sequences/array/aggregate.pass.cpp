//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure std::array is an aggregate type.

#include <array>
#include <type_traits>

template <typename T>
void tests()
{
    // Test aggregate initialization
    {
        std::array<T, 0> a0 = {}; (void)a0;
        std::array<T, 1> a1 = {T()}; (void)a1;
        std::array<T, 2> a2 = {T(), T()}; (void)a2;
        std::array<T, 3> a3 = {T(), T(), T()}; (void)a3;
    }

    // Test the is_aggregate trait.
#if TEST_STD_VER >= 17 // The trait is only available in C++17 and above
    static_assert(std::is_aggregate<std::array<T, 0> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 1> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 2> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 3> >::value, "");
    static_assert(std::is_aggregate<std::array<T, 4> >::value, "");
#endif
}

struct Empty { };
struct NonEmpty { int i; int j; };

int main(int, char**)
{
    tests<char>();
    tests<int>();
    tests<long>();
    tests<float>();
    tests<double>();
    tests<long double>();
    tests<NonEmpty>();
    tests<Empty>();

    return 0;
}
