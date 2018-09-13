
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// <set> feature macros

/*  Constant                                    Value
    __cpp_lib_allocator_traits_is_always_equal  201411L
    __cpp_lib_generic_associative_lookup        201304L
    __cpp_lib_node_extract                      201606L
    __cpp_lib_nonmember_container_access        201411L

*/

#include <set>
#include "test_macros.h"

int main()
{
//  ensure that the macros that are supposed to be defined in <set> are defined.

/*
#if !defined(__cpp_lib_fooby)
# error "__cpp_lib_fooby is not defined"
#elif __cpp_lib_fooby < 201606L
# error "__cpp_lib_fooby has an invalid value"
#endif
*/
}
