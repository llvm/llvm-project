/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include "tst-protected1mod.h"

int protected1 = -3;
int protected3 = -5;
static int expected_protected3 = -5;

asm (".protected protected3");

void
set_protected1b (int i)
{
  protected1 = i;
}

int *
protected1b_p (void)
{
  return &protected1;
}

void
set_expected_protected3b (int i)
{
  expected_protected3 = i;
}

void
set_protected3b (int i)
{
  protected3 = i;
  set_expected_protected3b (i);
}

int
check_protected3b (void)
{
  return protected3 == expected_protected3;
}

int *
protected3b_p (void)
{
  return &protected3;
}
