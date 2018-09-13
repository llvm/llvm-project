
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// <atomic> feature macros

/*  Constant                                    Value
    __cpp_lib_atomic_is_always_lock_free        201603L
    __cpp_lib_atomic_ref                        201806L

*/

#include <atomic>
#include "test_macros.h"

int main()
{
//  ensure that the macros that are supposed to be defined in <atomic> are defined.

#if _TEST_STD_VER > 14
# if !defined(__cpp_lib_atomic_is_always_lock_free)
#  error "__cpp_lib_atomic_is_always_lock_free is not defined"
# elif __cpp_lib_atomic_is_always_lock_free < 201603L
#  error "__cpp_lib_atomic_is_always_lock_free has an invalid value"
# endif
#endif

/*
#if !defined(__cpp_lib_fooby)
# error "__cpp_lib_fooby is not defined"
#elif __cpp_lib_fooby < 201606L
# error "__cpp_lib_fooby has an invalid value"
#endif
*/
}
