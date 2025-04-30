/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/ >.  */

#include <string.h>

#define STRCAT __strcat_power8

#undef libc_hidden_def
#define libc_hidden_def(name)

extern typeof (strcpy) __strcpy_power8;
extern typeof (strlen) __strlen_power8;

#define strcpy __strcpy_power8
#define strlen __strlen_power8
#include <string/strcat.c>
