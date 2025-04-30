/* Test size of the static TLS surplus reservation for backwards compatibility.
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

#include <stdio.h>
#include <support/check.h>
#include <support/xdlfcn.h>

static int do_test (void);
#include <support/test-driver.c>

/* This hack results in a definition of struct rtld_global_ro.  Do
   this after all the other header inclusions, to minimize the
   impact.  */
#define SHARED
#include <ldsodefs.h>

static
int do_test (void)
{
  /* Avoid introducing a copy relocation due to the hidden alias in
     ld.so.  */
  struct rtld_global_ro *glro = xdlsym (NULL, "_rtld_global_ro");
  printf ("info: _dl_tls_static_surplus: %zu\n", glro->_dl_tls_static_surplus);
  /* Hisoric value: 16 * 100 + 64.  */
  TEST_VERIFY (glro->_dl_tls_static_surplus >= 1664);
  return 0;
}
