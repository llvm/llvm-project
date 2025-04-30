/* Verify that changing xmm registers in audit library won't affect
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
#include <emmintrin.h>

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

ElfW(Addr)
pltenter (ElfW(Sym) *sym, unsigned int ndx, uintptr_t *refcook,
	  uintptr_t *defcook, La_regs *regs, unsigned int *flags,
	  const char *symname, long int *framesizep)
{
  printf ("pltenter: symname=%s, st_value=%#lx, ndx=%u, flags=%u\n",
	  symname, (long int) sym->st_value, ndx, *flags);

  __m128i minusone = _mm_set1_epi32 (-1);

  if (strcmp (symname, "audit_test") == 0)
    {
      __m128i zero = _mm_setzero_si128 ();
      if (memcmp (&regs->lr_xmm[0], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[1], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[2], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[3], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[4], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[5], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[6], &zero, sizeof (zero))
	  || memcmp (&regs->lr_xmm[7], &zero, sizeof (zero)))
	abort ();

      for (int i = 0; i < 8; i++)
	regs->lr_xmm[i] = (La_x86_64_xmm) _mm_set1_epi32 (i + 1);

      *framesizep = 1024;
    }

  asm volatile ("movdqa %0, %%xmm0" : : "x" (minusone) : "xmm0" );
  asm volatile ("movdqa %0, %%xmm1" : : "x" (minusone) : "xmm1" );
  asm volatile ("movdqa %0, %%xmm2" : : "x" (minusone) : "xmm2" );
  asm volatile ("movdqa %0, %%xmm3" : : "x" (minusone) : "xmm3" );
  asm volatile ("movdqa %0, %%xmm4" : : "x" (minusone) : "xmm4" );
  asm volatile ("movdqa %0, %%xmm5" : : "x" (minusone) : "xmm5" );
  asm volatile ("movdqa %0, %%xmm6" : : "x" (minusone) : "xmm6" );
  asm volatile ("movdqa %0, %%xmm7" : : "x" (minusone) : "xmm7" );

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

  __m128i xmm;

  if (strcmp (symname, "audit_test") == 0)
    {
      __m128i zero = _mm_setzero_si128 ();
      if (memcmp (&outregs->lrv_xmm0, &zero, sizeof (zero)))
	abort ();

      for (int i = 0; i < 8; i++)
	{
	  xmm = _mm_set1_epi32 (i + 1);
	  if (memcmp (&inregs->lr_xmm[i], &xmm, sizeof (xmm)) != 0)
	    abort ();
	}

      outregs->lrv_xmm0 = (La_x86_64_xmm) _mm_set1_epi32 (0x12349876);
    }

  xmm = _mm_set1_epi32 (-1);
  asm volatile ("movdqa %0, %%xmm0" : : "x" (xmm) : "xmm0" );
  asm volatile ("movdqa %0, %%xmm1" : : "x" (xmm) : "xmm1" );

  return 0;
}
