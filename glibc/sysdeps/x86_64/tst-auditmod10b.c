/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Verify that changing AVX512 registers in audit library won't affect
   function parameter passing/return.  */

#include <dlfcn.h>
#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <bits/wordsize.h>
#include <gnu/lib-names.h>

unsigned int
la_version (unsigned int v)
{
  setlinebuf (stdout);

  printf ("version: %u\n", v);

  char buf[20];
  sprintf (buf, "%u", v);

  return v;
}

void
la_activity (uintptr_t *cookie, unsigned int flag)
{
  if (flag == LA_ACT_CONSISTENT)
    printf ("activity: consistent\n");
  else if (flag == LA_ACT_ADD)
    printf ("activity: add\n");
  else if (flag == LA_ACT_DELETE)
    printf ("activity: delete\n");
  else
    printf ("activity: unknown activity %u\n", flag);
}

char *
la_objsearch (const char *name, uintptr_t *cookie, unsigned int flag)
{
  char buf[100];
  const char *flagstr;
  if (flag == LA_SER_ORIG)
    flagstr = "LA_SET_ORIG";
  else if (flag == LA_SER_LIBPATH)
    flagstr = "LA_SER_LIBPATH";
  else if (flag == LA_SER_RUNPATH)
    flagstr = "LA_SER_RUNPATH";
  else if (flag == LA_SER_CONFIG)
    flagstr = "LA_SER_CONFIG";
  else if (flag == LA_SER_DEFAULT)
    flagstr = "LA_SER_DEFAULT";
  else if (flag == LA_SER_SECURE)
    flagstr = "LA_SER_SECURE";
  else
    {
       sprintf (buf, "unknown flag %d", flag);
       flagstr = buf;
    }
  printf ("objsearch: %s, %s\n", name, flagstr);

  return (char *) name;
}

unsigned int
la_objopen (struct link_map *l, Lmid_t lmid, uintptr_t *cookie)
{
  printf ("objopen: %ld, %s\n", lmid, l->l_name);

  return 3;
}

void
la_preinit (uintptr_t *cookie)
{
  printf ("preinit\n");
}

unsigned int
la_objclose  (uintptr_t *cookie)
{
  printf ("objclose\n");
  return 0;
}

uintptr_t
la_symbind32 (Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
	      uintptr_t *defcook, unsigned int *flags, const char *symname)
{
  printf ("symbind32: symname=%s, st_value=%#lx, ndx=%u, flags=%u\n",
	  symname, (long int) sym->st_value, ndx, *flags);

  return sym->st_value;
}

uintptr_t
la_symbind64 (Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
	      uintptr_t *defcook, unsigned int *flags, const char *symname)
{
  printf ("symbind64: symname=%s, st_value=%#lx, ndx=%u, flags=%u\n",
	  symname, (long int) sym->st_value, ndx, *flags);

  return sym->st_value;
}

#include <tst-audit.h>

#ifdef __AVX512F__
#include <immintrin.h>
#include <cpuid.h>

static int
check_avx512 (void)
{
  unsigned int eax, ebx, ecx, edx;

  if (__get_cpuid (1, &eax, &ebx, &ecx, &edx) == 0
      || (ecx & (bit_AVX | bit_OSXSAVE)) != (bit_AVX | bit_OSXSAVE))
    return 0;

  __cpuid_count (7, 0, eax, ebx, ecx, edx);
  if (!(ebx & bit_AVX512F))
    return 0;

  asm ("xgetbv" : "=a" (eax), "=d" (edx) : "c" (0));

  /* Verify that ZMM, YMM and XMM states are enabled.  */
  return (eax & 0xe6) == 0xe6;
}

#else
#include <emmintrin.h>
#endif

ElfW(Addr)
pltenter (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
	  uintptr_t *defcook, La_regs *regs, unsigned int *flags,
	  const char *symname, long int *framesizep)
{
  printf ("pltenter: symname=%s, st_value=%#lx, ndx=%u, flags=%u\n",
	  symname, (long int) sym->st_value, ndx, *flags);

#ifdef __AVX512F__
  if (check_avx512 () && strcmp (symname, "audit_test") == 0)
    {
      __m512i zero = _mm512_setzero_si512 ();
      if (memcmp (&regs->lr_vector[0], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[1], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[2], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[3], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[4], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[5], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[6], &zero, sizeof (zero))
	  || memcmp (&regs->lr_vector[7], &zero, sizeof (zero)))
	abort ();

      for (int i = 0; i < 8; i++)
	regs->lr_vector[i].zmm[0]
	  = (La_x86_64_zmm) _mm512_set1_epi64 (i + 1);

      __m512i zmm = _mm512_set1_epi64 (-1);
      asm volatile ("vmovdqa64 %0, %%zmm0" : : "x" (zmm) : "xmm0" );
      asm volatile ("vmovdqa64 %0, %%zmm1" : : "x" (zmm) : "xmm1" );
      asm volatile ("vmovdqa64 %0, %%zmm2" : : "x" (zmm) : "xmm2" );
      asm volatile ("vmovdqa64 %0, %%zmm3" : : "x" (zmm) : "xmm3" );
      asm volatile ("vmovdqa64 %0, %%zmm4" : : "x" (zmm) : "xmm4" );
      asm volatile ("vmovdqa64 %0, %%zmm5" : : "x" (zmm) : "xmm5" );
      asm volatile ("vmovdqa64 %0, %%zmm6" : : "x" (zmm) : "xmm6" );
      asm volatile ("vmovdqa64 %0, %%zmm7" : : "x" (zmm) : "xmm7" );

      *framesizep = 1024;
    }
#endif

  return sym->st_value;
}

unsigned int
pltexit (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
	 uintptr_t *defcook, const La_regs *inregs, La_retval *outregs,
	 const char *symname)
{
  printf ("pltexit: symname=%s, st_value=%#lx, ndx=%u, retval=%tu\n",
	  symname, (long int) sym->st_value, ndx,
	  (ptrdiff_t) outregs->int_retval);

#ifdef __AVX512F__
  if (check_avx512 () && strcmp (symname, "audit_test") == 0)
    {
      __m512i zero = _mm512_setzero_si512 ();
      if (memcmp (&outregs->lrv_vector0, &zero, sizeof (zero)))
	abort ();

      for (int i = 0; i < 8; i++)
	{
	  __m512i zmm = _mm512_set1_epi64 (i + 1);
	  if (memcmp (&inregs->lr_vector[i], &zmm, sizeof (zmm)) != 0)
	    abort ();
	}

      outregs->lrv_vector0.zmm[0]
	= (La_x86_64_zmm) _mm512_set1_epi64 (0x12349876);

      __m512i zmm = _mm512_set1_epi64 (-1);
      asm volatile ("vmovdqa64 %0, %%zmm0" : : "x" (zmm) : "xmm0" );
      asm volatile ("vmovdqa64 %0, %%zmm1" : : "x" (zmm) : "xmm1" );
    }
#endif

  return 0;
}
