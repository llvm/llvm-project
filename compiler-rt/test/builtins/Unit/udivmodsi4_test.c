// RUN: %clang_builtins %s %librt -o %t && %run %t
//===-- udivmodsi4_test.c - Test __udivmodsi4 -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file tests __udivmodsi4 for the compiler_rt library.
//
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#include <stdio.h>

// Returns: a / b

extern COMPILER_RT_ABI su_int __udivmodsi4(su_int a, su_int b, su_int* rem);

int test__udivmodsi4(su_int a, su_int b, 
						su_int expected_result, su_int expected_rem)
{
	su_int rem;
    su_int result = __udivmodsi4(a, b, &rem);
    if (result != expected_result) {
        printf("error in __udivmodsi4: %u / %u = %u, expected %u\n",
               a, b, result, expected_result);
		return 1;
	}
    if (rem != expected_rem) {
        printf("error in __udivmodsi4: %u mod %u = %u, expected %u\n",
               a, b, rem, expected_rem);
		return 1;
	}
	
    return 0;
}


int main()
{
    if (test__udivmodsi4(0, 1, 0, 0))
        return 1;

    if (test__udivmodsi4(2, 1, 2, 0))
        return 1;

	if (test__udivmodsi4(19, 5, 3, 4))
        return 1;

	if (test__udivmodsi4(0x80000000, 8, 0x10000000, 0))
        return 1;
  	
 	if (test__udivmodsi4(0x80000003, 8, 0x10000000, 3))
        return 1;

	return 0;
}
