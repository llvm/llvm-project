/* Test DST expansion for static binaries doesn't carsh.  Bug 23462.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* The purpose of this test is to exercise the code in elf/dl-loac.c
   (_dl_init_paths) or thereabout and ensure that static binaries
   don't crash when expanding DSTs.

   If the dynamic loader code linked into the static binary cannot
   handle expanding the DSTs e.g. null-deref on an incomplete link
   map, then it will crash before reaching main, so the test harness
   is unnecessary.  */

int
main (void)
{
  return 0;
}
