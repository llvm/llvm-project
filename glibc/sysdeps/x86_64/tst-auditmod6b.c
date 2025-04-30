/* Verify that changing AVX registers in audit library won't affect
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

#ifdef __AVX__
#include <immintrin.h>
#include <cpuid.h>

static int avx = -1;

static inline int
__attribute ((always_inline))
check_avx (void)
{
  if (avx == -1)
    {
      unsigned int eax, ebx, ecx, edx;

      if (__get_cpuid (1, &eax, &ebx, &ecx, &edx)
	  && (ecx & bit_AVX))
	avx = 1;
      else
	avx = 0;
    }
  return avx;
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

#ifdef __AVX__
  if (check_avx () && strcmp (symname, "audit_test") == 0)
    {
      int i;

      __m128i xmm = _mm_setzero_si128 ();
      for (i = 0; i < 8; i++)
	if (memcmp (&regs->lr_xmm[i], &xmm, sizeof (xmm))
	    || memcmp (&regs->lr_vector[i], &xmm, sizeof (xmm)))
	  abort ();

      for (i = 0; i < 8; i += 2)
	{
	  regs->lr_xmm[i] = (La_x86_64_xmm) _mm_set1_epi32 (i + 1);
	  regs->lr_vector[i].xmm[0] = regs->lr_xmm[i];
	  regs->lr_vector[i + 1].ymm[0]
	    = (La_x86_64_ymm) _mm256_set1_epi32 (i + 2);
	  regs->lr_xmm[i + 1] = regs->lr_vector[i + 1].xmm[0];
	}

      __m256i ymm = _mm256_set1_epi32 (-1);
      asm volatile ("vmovdqa %0, %%ymm0" : : "x" (ymm) : "xmm0" );
      asm volatile ("vmovdqa %0, %%ymm1" : : "x" (ymm) : "xmm1" );
      asm volatile ("vmovdqa %0, %%ymm2" : : "x" (ymm) : "xmm2" );
      asm volatile ("vmovdqa %0, %%ymm3" : : "x" (ymm) : "xmm3" );
      asm volatile ("vmovdqa %0, %%ymm4" : : "x" (ymm) : "xmm4" );
      asm volatile ("vmovdqa %0, %%ymm5" : : "x" (ymm) : "xmm5" );
      asm volatile ("vmovdqa %0, %%ymm6" : : "x" (ymm) : "xmm6" );
      asm volatile ("vmovdqa %0, %%ymm7" : : "x" (ymm) : "xmm7" );

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

#ifdef __AVX__
  if (check_avx () && strcmp (symname, "audit_test") == 0)
    {
      int i;

      __m128i xmm = _mm_setzero_si128 ();
      if (memcmp (&outregs->lrv_xmm0, &xmm, sizeof (xmm))
	  || memcmp (&outregs->lrv_vector0, &xmm, sizeof (xmm)))
	abort ();

      __m256i ymm;

      for (i = 0; i < 8; i += 2)
	{
	  xmm = _mm_set1_epi32 (i + 0x100);
	  if (memcmp (&inregs->lr_xmm[i], &xmm, sizeof (xmm))
	      || memcmp (&inregs->lr_vector[i], &xmm, sizeof (xmm)))
	    abort ();

	  ymm = _mm256_set1_epi32 (i + 0x101);
	  if (memcmp (&inregs->lr_xmm[i + 1],
		      &inregs->lr_vector[i + 1].xmm[0], sizeof (xmm))
	      || memcmp (&inregs->lr_vector[i + 1], &ymm, sizeof (ymm)))
	    abort ();
	}

      outregs->lrv_vector0.ymm[0]
	= (La_x86_64_ymm) _mm256_set1_epi32 (0x12349876);

      ymm = _mm256_set1_epi32 (-1);
      asm volatile ("vmovdqa %0, %%ymm0" : : "x" (ymm) : "xmm0" );
      asm volatile ("vmovdqa %0, %%ymm1" : : "x" (ymm) : "xmm1" );
    }
#endif

  return 0;
}
