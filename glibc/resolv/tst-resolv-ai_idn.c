/* Test getaddrinfo and getnameinfo with AI_IDN, NI_IDN (UTF-8).
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#define TEST_USE_UTF8 1
#include "tst-resolv-ai_idn-common.c"

#include <locale.h>
#include <support/xdlfcn.h>

static int
do_test (void)
{
  void *handle = dlopen (LIBIDN2_SONAME, RTLD_LAZY);
  if (handle == NULL)
    FAIL_UNSUPPORTED ("libidn2 not installed");
  void *check_ver_sym = xdlsym (handle, "idn2_check_version");
  const char *check_res
    = ((const char *(*) (const char *)) check_ver_sym) ("2.0.5");
  if (check_res == NULL)
    FAIL_UNSUPPORTED ("libidn2 too old");

  if (setlocale (LC_CTYPE, "en_US.UTF-8") == NULL)
    FAIL_EXIT1 ("setlocale: %m");

  struct resolv_test *aux = resolv_test_start
    ((struct resolv_redirect_config)
     {
       .response_callback = response,
     });

  gai_tests_with_libidn2 ();
  gni_tests_with_libidn2 ();

  resolv_test_end (aux);
  xdlclose (handle);
  return 0;
}

#include <support/test-driver.c>
