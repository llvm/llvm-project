//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <ios>
//
// class ios_base
// {
// public:
//     typedef OFF_T streamoff;
// };

#include <ios>
#include <type_traits>

int main()
{
    static_assert((std::is_integral<std::ios_base::streamoff>::value), "");
    static_assert((std::is_signed<std::ios_base::streamoff>::value), "");
}
