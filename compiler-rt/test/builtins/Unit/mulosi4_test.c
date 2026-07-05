// RUN: %clang_builtins %s %librt -o %t && %run %t
// REQUIRES: librt_has_mulosi4

#include "int_lib.h"
#include <stdio.h>

// Returns: a * b

// Effects: aborts if a * b overflows

COMPILER_RT_ABI si_int __mulosi4(si_int a, si_int b, int *overflow);

int test__mulosi4(si_int a, si_int b, si_int expected, int expected_overflow)
{
  int ov;
  si_int x = __mulosi4(a, b, &ov);
  if (ov != expected_overflow)
    printf("error in __mulosi4: overflow=%d expected=%d\n",
	   ov, expected_overflow);
  else if (!expected_overflow && x != expected) {
    printf("error in __mulosi4: 0x%X * 0x%X = 0x%X (overflow=%d), "
	   "expected 0x%X (overflow=%d)\n",
	   a, b, x, ov, expected, expected_overflow);
    return 1;
  }
  return 0;
}


int main()
{
    if (test__mulosi4(0, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, 1, 0, 0))
        return 1;
    if (test__mulosi4(1, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, 10, 0, 0))
        return 1;
    if (test__mulosi4(10, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, 0x1234567, 0, 0))
        return 1;
    if (test__mulosi4(0x1234567, 0, 0, 0))
        return 1;

    if (test__mulosi4(0, -1, 0, 0))
        return 1;
    if (test__mulosi4(-1, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, -10, 0, 0))
        return 1;
    if (test__mulosi4(-10, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, -0x1234567, 0, 0))
        return 1;
    if (test__mulosi4(-0x1234567, 0, 0, 0))
        return 1;

    if (test__mulosi4(1, 1, 1, 0))
        return 1;
    if (test__mulosi4(1, 10, 10, 0))
        return 1;
    if (test__mulosi4(10, 1, 10, 0))
        return 1;
    if (test__mulosi4(1, 0x1234567, 0x1234567, 0))
        return 1;
    if (test__mulosi4(0x1234567, 1, 0x1234567, 0))
        return 1;

    if (test__mulosi4(1, -1, -1, 0))
        return 1;
    if (test__mulosi4(1, -10, -10, 0))
        return 1;
    if (test__mulosi4(-10, 1, -10, 0))
        return 1;
    if (test__mulosi4(1, -0x1234567, -0x1234567, 0))
        return 1;
    if (test__mulosi4(-0x1234567, 1, -0x1234567, 0))
        return 1;

     if (test__mulosi4(0x7FFFFFFF, -2, 0x80000001, 1))
         return 1;
     if (test__mulosi4(-2, 0x7FFFFFFF, 0x80000001, 1))
         return 1;
    if (test__mulosi4(0x7FFFFFFF, -1, 0x80000001, 0))
        return 1;
    if (test__mulosi4(-1, 0x7FFFFFFF, 0x80000001, 0))
        return 1;
    if (test__mulosi4(0x7FFFFFFF, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, 0x7FFFFFFF, 0, 0))
        return 1;
    if (test__mulosi4(0x7FFFFFFF, 1, 0x7FFFFFFF, 0))
        return 1;
    if (test__mulosi4(1, 0x7FFFFFFF, 0x7FFFFFFF, 0))
        return 1;
     if (test__mulosi4(0x7FFFFFFF, 2, 0x80000001, 1))
         return 1;
     if (test__mulosi4(2, 0x7FFFFFFF, 0x80000001, 1))
         return 1;

     if (test__mulosi4(0x80000000, -2, 0x80000000, 1))
         return 1;
     if (test__mulosi4(-2, 0x80000000, 0x80000000, 1))
         return 1;
     if (test__mulosi4(0x80000000, -1, 0x80000000, 1))
         return 1;
     if (test__mulosi4(-1, 0x80000000, 0x80000000, 1))
         return 1;
    if (test__mulosi4(0x80000000, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, 0x80000000, 0, 0))
        return 1;
    if (test__mulosi4(0x80000000, 1, 0x80000000, 0))
        return 1;
    if (test__mulosi4(1, 0x80000000, 0x80000000, 0))
        return 1;
     if (test__mulosi4(0x80000000, 2, 0x80000000, 1))
         return 1;
     if (test__mulosi4(2, 0x80000000, 0x80000000, 1))
         return 1;

     if (test__mulosi4(0x80000001, -2, 0x80000001, 1))
         return 1;
     if (test__mulosi4(-2, 0x80000001, 0x80000001, 1))
         return 1;
    if (test__mulosi4(0x80000001, -1, 0x7FFFFFFF, 0))
        return 1;
    if (test__mulosi4(-1, 0x80000001, 0x7FFFFFFF, 0))
        return 1;
    if (test__mulosi4(0x80000001, 0, 0, 0))
        return 1;
    if (test__mulosi4(0, 0x80000001, 0, 0))
        return 1;
    if (test__mulosi4(0x80000001, 1, 0x80000001, 0))
        return 1;
    if (test__mulosi4(1, 0x80000001, 0x80000001, 0))
        return 1;
     if (test__mulosi4(0x80000001, 2, 0x80000000, 1))
         return 1;
     if (test__mulosi4(2, 0x80000001, 0x80000000, 1))
         return 1;

    return 0;
}
