/* Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/ >.  */

#include <string.h>

#define STRNCAT __strncat_power8

extern __typeof (strncat) __strncat_power8 attribute_hidden;
extern __typeof (strlen) __strlen_power8 attribute_hidden;
extern __typeof (strnlen) __strnlen_power8 attribute_hidden;
extern __typeof (memcpy) __memcpy_power7 attribute_hidden;

#define strlen    __strlen_power8
#define __strnlen __strnlen_power8
#define memcpy    __memcpy_power7

#include <string/strncat.c>
