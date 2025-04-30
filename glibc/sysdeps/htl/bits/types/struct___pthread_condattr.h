/* Condition attribute type.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _BITS_TYPES_STRUCT___PTHREAD_CONDATTR
#define _BITS_TYPES_STRUCT___PTHREAD_CONDATTR	1

#include <bits/types.h>

enum __pthread_process_shared;

/* User visible part of a condition attribute variable.  */
struct __pthread_condattr
{
  enum __pthread_process_shared __pshared;
  __clockid_t __clock;
};

#endif /* bits/types/struct___pthread_condattr.h */
