/* Test program for the gcc interface.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <stdio.h>

#define __no_type_class		-1
#define __void_type_class	 0
#define __integer_type_class	 1
#define __char_type_class	 2
#define __enumeral_type_class	 3
#define __boolean_type_class	 4
#define __pointer_type_class	 5
#define __reference_type_class	 6
#define __offset_type_class	 7
#define __real_type_class	 8
#define __complex_type_class	 9
#define __function_type_class	10
#define __method_type_class	11
#define __record_type_class	12
#define __union_type_class	13
#define __array_type_class	14
#define __string_type_class	15
#define __set_type_class	16
#define __file_type_class	17
#define __lang_type_class	18


#define TEST(var) \
  ({ int wrong = (__builtin_classify_type (__##var##_type)		      \
		  != __##var##_type_class);				      \
     printf ("%-15s is %d: %s\n",					      \
	     #var, __builtin_classify_type (__##var##_type),		      \
	     wrong ? "WRONG" : "OK");					      \
     wrong;								      \
  })


static int
do_test (void)
{
  int result = 0;
  int __integer_type;
  void *__pointer_type;
  double __real_type;
  __complex__ double __complex_type;
  struct { int a; } __record_type;
  union { int a; int b; } __union_type;

  result |= TEST (integer);
  result |= TEST (pointer);
  result |= TEST (real);
  result |= TEST (complex);
  result |= TEST (record);
  result |= TEST (union);

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
