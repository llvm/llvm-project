/* Test string/memory functions with size_t in the lower 32 bits of
   64-bit register.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

#define TEST_MAIN
#include <string/test-string.h>

/* On x32, parameter_t may be passed in a 64-bit register with the LEN
   field in the lower 32 bits.  When the LEN field of 64-bit register
   is passed to string/memory function as the size_t parameter, only
   the lower 32 bits can be used.  */
typedef struct
{
  union
    {
      size_t len;
      void (*fn) (void);
    };
  void *p;
} parameter_t;
