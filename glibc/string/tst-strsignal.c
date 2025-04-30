/* Test for strsignal.

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

#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <locale.h>
#include <array_length.h>

#include <support/support.h>
#include <support/check.h>

static int
do_test (void)
{
  unsetenv ("LANGUAGE");

  xsetlocale (LC_ALL, "C");

  TEST_COMPARE_STRING (strsignal (SIGINT),     "Interrupt");
  TEST_COMPARE_STRING (strsignal (-1),         "Unknown signal -1");
#ifdef SIGRTMIN
  if (SIGRTMIN < SIGRTMAX)
    TEST_COMPARE_STRING (strsignal (SIGRTMIN),   "Real-time signal 0");
#endif
#ifdef SIGRTMAX
  if (SIGRTMAX == 64)
    TEST_COMPARE_STRING (strsignal (SIGRTMAX+1), "Unknown signal 65");
#endif

  xsetlocale (LC_ALL, "pt_BR.UTF-8");

  TEST_COMPARE_STRING (strsignal (SIGINT),    "Interrup\xc3\xa7\xc3\xa3\x6f");
  TEST_COMPARE_STRING (strsignal (-1),        "Sinal desconhecido -1");
#ifdef SIGRTMI
  if (SIGRTMIN < SIGRTMAX)
    TEST_COMPARE_STRING (strsignal (SIGRTMIN),  "Sinal de tempo-real 0");
#endif
#ifdef SIGRTMAX
  if (SIGRTMAX == 64)
    TEST_COMPARE_STRING (strsignal (SIGRTMAX+1), "Sinal desconhecido 65");
#endif

  return 0;
}

#include <support/test-driver.c>
