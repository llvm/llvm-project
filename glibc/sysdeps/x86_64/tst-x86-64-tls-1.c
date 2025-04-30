/* Test THREAD_SETMEM and THREAD_SETMEM_NC for IMM64.
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

#include <tls.h>
#include <support/check.h>

static int
do_test (void)
{
  unsigned long long int saved_ssp_base, ssp_base;
  saved_ssp_base = THREAD_GETMEM (THREAD_SELF, header.ssp_base);

  THREAD_SETMEM (THREAD_SELF, header.ssp_base, (1ULL << 57) - 1);
  ssp_base = THREAD_GETMEM (THREAD_SELF, header.ssp_base);
  if (ssp_base != ((1ULL << 57) - 1))
    FAIL_EXIT1 ("THREAD_SETMEM: 0x%llx != 0x%llx",
		ssp_base, (1ULL << 57) - 1);

  THREAD_SETMEM (THREAD_SELF, header.ssp_base, -1ULL);
  ssp_base = THREAD_GETMEM (THREAD_SELF, header.ssp_base);
  if (ssp_base != -1ULL)
    FAIL_EXIT1 ("THREAD_SETMEM: 0x%llx != 0x%llx", ssp_base, -1ULL);

  THREAD_SETMEM (THREAD_SELF, header.ssp_base, saved_ssp_base);
#ifndef __ILP32__
  struct pthread_key_data *saved_specific, *specific;
  saved_specific = THREAD_GETMEM_NC (THREAD_SELF, specific, 1);

  uintptr_t value = (1UL << 57) - 1;
  THREAD_SETMEM_NC (THREAD_SELF, specific, 1,
		    (struct pthread_key_data *) value);
  specific = THREAD_GETMEM_NC (THREAD_SELF, specific, 1);
  if (specific != (struct pthread_key_data *) value)
    FAIL_EXIT1 ("THREAD_GETMEM_NC: %p != %p",
		specific, (struct pthread_key_data *) value);

  THREAD_SETMEM_NC (THREAD_SELF, specific, 1,
		    (struct pthread_key_data *) -1UL);
  specific = THREAD_GETMEM_NC (THREAD_SELF, specific, 1);
  if (specific != (struct pthread_key_data *) -1UL)
    FAIL_EXIT1 ("THREAD_GETMEM_NC: %p != %p",
		specific, (struct pthread_key_data *) -1UL);

  THREAD_SETMEM_NC (THREAD_SELF, specific, 1, saved_specific);
#endif
  return 0;
}

#include <support/test-driver.c>
