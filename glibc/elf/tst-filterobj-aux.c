/* Auxiliary filter object.
   Contains symbols to be resolved in filtee, and one which doesn't.

   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include "tst-filterobj-filtee.h"

/* We never want to see the output of the auxiliary object.  */
const char *get_text (void)
{
  return "Hello from auxiliary filter object (FAIL)";
}

/* The filtee doesn't implement this symbol, so this should resolve.  */
const char *get_text2 (void)
{
  return "Hello from auxiliary filter object (PASS)";
}
