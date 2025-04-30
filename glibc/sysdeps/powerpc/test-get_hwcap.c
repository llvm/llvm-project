/* Check __ppc_get_hwcap() and __ppc_get_at_plaftorm() functionality.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Tests if the hwcap, hwcap2 and platform data are stored in the TCB.  */

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>
#include <pthread.h>

#include <support/check.h>
#include <support/xthread.h>

#include <sys/auxv.h>

#include <dl-procinfo.h>

#ifndef STATIC_TST_HWCAP
#undef PROCINFO_DECL
#include <dl-procinfo.c>
#endif

/* Offsets copied from tcb-offsets.h.  */

#ifdef __powerpc64__
# define __TPREG     "r13"
# define __HWCAPOFF -28776
# define __ATPLATOFF -28764
#else
# define __TPREG     "r2"
# define __HWCAPOFF -28736
# define __HWCAP2OFF -28732
# define __ATPLATOFF -28724
#endif

uint64_t check_tcbhwcap (long tid)
{

  uint32_t tcb_at_platform, at_platform;
  uint64_t hwcap, hwcap2, tcb_hwcap;
  const char *at_platform_string;

  /* Testing if the hwcap/hwcap2 data is correctly initialized by
     TLS_TP_INIT.  */

  register unsigned long __tp __asm__ (__TPREG);

#ifdef __powerpc64__
  __asm__  ("ld %0,%1(%2)\n"
	    : "=r" (tcb_hwcap)
	    : "i" (__HWCAPOFF), "b" (__tp));
#else
  uint64_t h1, h2;

  __asm__ ("lwz %0,%1(%2)\n"
      : "=r" (h1)
      : "i" (__HWCAPOFF), "b" (__tp));
  __asm__ ("lwz %0,%1(%2)\n"
      : "=r" (h2)
      : "i" (__HWCAP2OFF), "b" (__tp));
  tcb_hwcap = (h1 >> 32) << 32 | (h2 >> 32);
#endif

  hwcap = getauxval (AT_HWCAP);
  hwcap2 = getauxval (AT_HWCAP2);

  /* hwcap contains only the latest supported ISA, the code checks which is
     and fills the previous supported ones.  This is necessary because the
     same is done in hwcapinfo.c when setting the values that are copied to
     the TCB.  */

  if (hwcap2 & PPC_FEATURE2_ARCH_2_07)
    hwcap |= PPC_FEATURE_ARCH_2_06
	  | PPC_FEATURE_ARCH_2_05
	  | PPC_FEATURE_POWER5_PLUS
	  | PPC_FEATURE_POWER5
	  | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_ARCH_2_06)
    hwcap |= PPC_FEATURE_ARCH_2_05
	  | PPC_FEATURE_POWER5_PLUS
	  | PPC_FEATURE_POWER5
	  | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_ARCH_2_05)
    hwcap |= PPC_FEATURE_POWER5_PLUS
	  | PPC_FEATURE_POWER5
	  | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_POWER5_PLUS)
    hwcap |= PPC_FEATURE_POWER5
	  | PPC_FEATURE_POWER4;
  else if (hwcap & PPC_FEATURE_POWER5)
    hwcap |= PPC_FEATURE_POWER4;

  hwcap = (hwcap << 32) + hwcap2;

  if ( tcb_hwcap != hwcap )
    {
      printf ("FAIL: __ppc_get_hwcap() - HWCAP is %" PRIx64 ". Should be %"
	      PRIx64 " for thread %ld.\n", tcb_hwcap, hwcap, tid);
      return 1;
    }

  /* Same test for the platform number.  */
  __asm__  ("lwz %0,%1(%2)\n"
	    : "=r" (tcb_at_platform)
	    : "i" (__ATPLATOFF), "b" (__tp));

  at_platform_string = (const char *) getauxval (AT_PLATFORM);
  at_platform = _dl_string_platform (at_platform_string);

  if ( tcb_at_platform != at_platform )
    {
      printf ("FAIL: __ppc_get_at_platform() - AT_PLATFORM is %x. Should be %x"
	     " for thread %ld\n", tcb_at_platform, at_platform, tid);
      return 1;
    }

  return 0;
}

void *t1 (void *tid)
{
  if (check_tcbhwcap ((long) tid))
    {
      pthread_exit (tid);
    }

  pthread_exit (NULL);

}

static int
do_test (void)
{

  pthread_t threads[2];
  pthread_attr_t attr;
  pthread_attr_init (&attr);
  pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_JOINABLE);

  long i = 0;

  /* Check for main.  */
  if (check_tcbhwcap (i))
    {
      return 1;
    }

  /* Check for other thread.  */
  i++;
  threads[i] = xpthread_create (&attr, t1, (void *)i);

  pthread_attr_destroy (&attr);
  TEST_VERIFY_EXIT (xpthread_join (threads[i]) == NULL);

  printf("PASS: HWCAP, HWCAP2 and AT_PLATFORM are correctly set in the TCB for"
	 " all threads.\n");

  pthread_exit (NULL);

}

#include <support/test-driver.c>
