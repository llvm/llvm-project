/* Module for test with many dynamic TLS variables.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* This file is parameterized by macros NAME, SETTER, GETTER, which
   are set form the Makefile.  */

#include "tst-tls-manydynamic.h"

__thread struct value NAME;

void
SETTER (const struct value *value)
{
  NAME = *value;
}

void
GETTER (struct value *value)
{
  *value = NAME;
}
