//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <version>
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7
// UNSUPPORTED: clang-3.8, clang-3.9, clang-4.0, clang-5.0, clang-6.0

#include <version>

#if !defined(_LIBCPP_VERSION)
#error "_LIBCPP_VERSION must be defined after including <version>"
#endif

int main()
{
}
