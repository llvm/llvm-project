//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_extent

#include <type_traits>

enum Enum {zero, one_};

int main()
{
    static_assert((std::is_same<std::remove_extent<int>::type, int>::value), "");
    static_assert((std::is_same<std::remove_extent<const Enum>::type, const Enum>::value), "");
    static_assert((std::is_same<std::remove_extent<int[]>::type, int>::value), "");
    static_assert((std::is_same<std::remove_extent<const int[]>::type, const int>::value), "");
    static_assert((std::is_same<std::remove_extent<int[3]>::type, int>::value), "");
    static_assert((std::is_same<std::remove_extent<const int[3]>::type, const int>::value), "");
    static_assert((std::is_same<std::remove_extent<int[][3]>::type, int[3]>::value), "");
    static_assert((std::is_same<std::remove_extent<const int[][3]>::type, const int[3]>::value), "");
    static_assert((std::is_same<std::remove_extent<int[2][3]>::type, int[3]>::value), "");
    static_assert((std::is_same<std::remove_extent<const int[2][3]>::type, const int[3]>::value), "");
    static_assert((std::is_same<std::remove_extent<int[1][2][3]>::type, int[2][3]>::value), "");
    static_assert((std::is_same<std::remove_extent<const int[1][2][3]>::type, const int[2][3]>::value), "");
}
