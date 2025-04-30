/* Test NPTL with stack limit that is not a multiple of the page size.
   HPPA version.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* This sets the stack resource limit to 8193kb, which is not a multiple
   of the page size, and therefore an odd sized stack limit.  We override
   this because the default is too small to run with.  */

#define ODD_STACK_LIMIT (8193 * 1024)

#include <sysdeps/../nptl/tst-oddstacklimit.c>
