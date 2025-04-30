/* Exercise dlerror_run in elf/dl-libc.c after dlmopen, via NSS (bug 27646).
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

#include <support/xdlfcn.h>

static int
do_test (void)
{
  void *handle = xdlmopen (LM_ID_NEWLM, "tst-dlmopen-gethostbyname-mod.so",
                           RTLD_NOW);
  void (*call_gethostbyname) (void) = xdlsym (handle, "call_gethostbyname");
  call_gethostbyname ();
  return 0;
}

#include <support/test-driver.c>
