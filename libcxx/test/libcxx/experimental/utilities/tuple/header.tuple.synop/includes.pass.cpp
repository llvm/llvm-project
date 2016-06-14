//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <experimental/tuple>

#include <experimental/tuple>

int main()
{
#ifndef _LIBCPP_TUPLE
#  error "<experimental/tuple> must include <tuple>"
#endif
}
