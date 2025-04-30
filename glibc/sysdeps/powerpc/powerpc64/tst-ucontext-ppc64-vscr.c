/* Test if POWER vscr read by ucontext.
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

#include <support/check.h>
#include <sys/auxv.h>
#include <ucontext.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <altivec.h>

#define SAT 0x1

/* This test is supported only on POWER 5 or higher.  */
#define PPC_CPU_SUPPORTED (PPC_FEATURE_POWER5 | PPC_FEATURE_POWER5_PLUS \
			   | PPC_FEATURE_ARCH_2_05 | PPC_FEATURE_ARCH_2_06 \
			   | PPC_FEATURE2_ARCH_2_07)
static int
do_test (void)
{

  if (!(getauxval(AT_HWCAP2) & PPC_CPU_SUPPORTED))
    {
      if (!(getauxval(AT_HWCAP) & PPC_CPU_SUPPORTED))
      FAIL_UNSUPPORTED("This test is unsupported on POWER < 5\n");
    }

  uint32_t vscr[4] __attribute__ ((aligned (16)));
  uint32_t* vscr_ptr = vscr;
  uint32_t vscr_word;
  ucontext_t ucp;
  __vector unsigned int v0 = {0};
  __vector unsigned int v1 = {0};

  /* Set SAT bit in VSCR register.  */
  asm volatile (".machine push;\n"
		".machine \"power5\";\n"
		"vspltisb %0,0;\n"
		"vspltisb %1,-1;\n"
		"vpkuwus %0,%0,%1;\n"
		"mfvscr %0;\n"
		"stvx %0,0,%2;\n"
		".machine pop;"
		: "=v" (v0), "=v" (v1)
		: "r" (vscr_ptr)
		: "memory");
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  vscr_word = vscr[0];
#else
  vscr_word = vscr[3];
#endif

  if ((vscr_word & SAT) != SAT)
    {
      FAIL_EXIT1("FAIL: SAT bit is not set.\n");
    }

  if (getcontext (&ucp))
    {
      FAIL_EXIT1("FAIL: getcontext error\n");
    }
  if (ucp.uc_mcontext.v_regs->vscr.vscr_word != vscr_word)
    {
      FAIL_EXIT1("FAIL: ucontext vscr does not match with vscr\n");
    }
  return 0;
}

#include <support/test-driver.c>
