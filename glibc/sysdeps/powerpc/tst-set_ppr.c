/* Test the implementation of __ppc_set_ppr_* functions.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/auxv.h>

#include <sys/platform/ppc.h>

#include <support/test-driver.h>

#ifdef __powerpc64__
  typedef uint64_t ppr_t;
# define MFPPR "mfppr"
  /* The thread priority value is obtained from bits 11:13.  */
# define EXTRACT_THREAD_PRIORITY(x) ((x >> 50) & 7)
#else
  typedef uint32_t ppr_t;
# define MFPPR "mfppr32"
  /* For 32-bit, the upper 32 bits of the Program Priority Register (PPR)
     are used, so the thread priority value is obtained from bits 43:46.  */
# define EXTRACT_THREAD_PRIORITY(x) ((x >> 18) & 7)
#endif /* !__powerpc64__ */

/* Read the thread priority value set in the PPR.  */
static __inline__ ppr_t
get_thread_priority (void)
{
  /* Read the PPR.  */
  ppr_t ppr;
  asm volatile (MFPPR" %0" : "=r"(ppr));
  /* Return the thread priority value.  */
  return EXTRACT_THREAD_PRIORITY (ppr);
}

/* Check the thread priority bits of PPR are set as expected. */
uint8_t
check_thread_priority (uint8_t expected)
{
  ppr_t actual = get_thread_priority ();

  if (actual != expected)
    {
      printf ("FAIL: Expected %"PRIu8" got %"PRIuMAX".\n", expected,
	      (uintmax_t) actual);
      return 1;
    }
  printf ("PASS: Thread priority set to %"PRIu8" correctly.\n", expected);
  return 0;
}

/* The Power ISA 2.06 allows the following thread priorities for any
   problem state program: low (2), medium low (3), and medium (4).
   Power ISA 2.07b added very low (1).
   Check whether the values set by __ppc_set_ppr_* are correct.  */
static int
do_test (void)
{
  /* Check for the minimum required Power ISA to run these tests.  */
  if ((getauxval (AT_HWCAP) & PPC_FEATURE_ARCH_2_06) == 0)
    {
      printf ("Requires an environment that implements the Power ISA version"
              " 2.06 or greater.\n");
      return EXIT_UNSUPPORTED;
    }

  uint8_t rc = 0;

#ifdef _ARCH_PWR8
  __ppc_set_ppr_very_low ();
  rc |= check_thread_priority (1);
#endif /* _ARCH_PWR8 */

  __ppc_set_ppr_low ();
  rc |= check_thread_priority (2);

  __ppc_set_ppr_med_low ();
  rc |= check_thread_priority (3);

  __ppc_set_ppr_med ();
  rc |= check_thread_priority (4);

  return rc;
}

#include <support/test-driver.c>
