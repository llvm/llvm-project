/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Geoffrey Keating <Geoff.Keating@anu.edu.au>, 1997.

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
#include <math.h>
#include <gmp.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <stdlib.h>

#define PRINT_ERRORS 0

#define TOL 80
#define N2 18
#define FRAC (32*4)

#define mpbpl (CHAR_BIT * sizeof (mp_limb_t))
#define SZ (FRAC / mpbpl + 1)
typedef mp_limb_t mp1[SZ], mp2[SZ * 2];

#if BITS_PER_MP_LIMB == 64
# define LIMB64(L, H) 0x ## H ## L
#elif BITS_PER_MP_LIMB == 32
# define LIMB64(L, H) 0x ## L, 0x ## H
#else
# error
#endif

/* Once upon a time these constants were generated to 400 bits.
   We only need FRAC bits (128) at present, but we retain 384 bits
   in the text Just In Case.  */
#define CONSTSZ(INT, F1, F2, F3, F4, F5, F6, F7, F8, F9, Fa, Fb, Fc) \
	LIMB64(F4, F3), LIMB64(F2, F1), INT

static const mp1 mp_exp1 = {
  CONSTSZ (2, b7e15162, 8aed2a6a, bf715880, 9cf4f3c7, 62e7160f, 38b4da56,
           a784d904, 5190cfef, 324e7738, 926cfbe5, f4bf8d8d, 8c31d763)
};

static const mp1 mp_log2 = {
  CONSTSZ (0, b17217f7, d1cf79ab, c9e3b398, 03f2f6af, 40f34326, 7298b62d,
           8a0d175b, 8baafa2b, e7b87620, 6debac98, 559552fb, 4afa1b10)
};

static void
print_mpn_fp (const mp_limb_t *x, unsigned int dp, unsigned int base)
{
   static const char hexdig[16] = "0123456789abcdef";
   unsigned int i;
   mp1 tx;

   memcpy (tx, x, sizeof (mp1));
   if (base == 16)
     fputs ("0x", stdout);
   assert (x[SZ-1] < base);
   fputc (hexdig[x[SZ - 1]], stdout);
   fputc ('.', stdout);
   for (i = 0; i < dp; i++)
     {
       tx[SZ - 1] = 0;
       mpn_mul_1 (tx, tx, SZ, base);
       assert (tx[SZ - 1] < base);
       fputc (hexdig[tx[SZ - 1]], stdout);
     }
}

/* Compute e^x.  */
static void
exp_mpn (mp1 ex, mp1 x)
{
   unsigned int n;
   mp1 xp;
   mp2 tmp;
   mp_limb_t chk __attribute__ ((unused));
   mp1 tol;

   memset (xp, 0, sizeof (mp1));
   memset (ex, 0, sizeof (mp1));
   xp[FRAC / mpbpl] = (mp_limb_t)1 << FRAC % mpbpl;
   memset (tol, 0, sizeof (mp1));
   tol[(FRAC - TOL) / mpbpl] = (mp_limb_t)1 << (FRAC - TOL) % mpbpl;

   n = 0;

   do
     {
       /* Calculate sum(x^n/n!) until the next term is sufficiently small.  */

       mpn_mul_n (tmp, xp, x, SZ);
       assert(tmp[SZ * 2 - 1] == 0);
       if (n > 0)
	 mpn_divmod_1 (xp, tmp + FRAC / mpbpl, SZ, n);
       chk = mpn_add_n (ex, ex, xp, SZ);
       assert (chk == 0);
       ++n;
       assert (n < 80); /* Catch too-high TOL.  */
     }
   while (n < 10 || mpn_cmp (xp, tol, SZ) >= 0);
}

/* Calculate 2^x.  */
static void
exp2_mpn (mp1 ex, mp1 x)
{
  mp2 tmp;
  mpn_mul_n (tmp, x, mp_log2, SZ);
  assert(tmp[SZ * 2 - 1] == 0);
  exp_mpn (ex, tmp + FRAC / mpbpl);
}


static int
mpn_bitsize(const mp_limb_t *SRC_PTR, mp_size_t SIZE)
{
  int i, j;
  for (i = SIZE - 1; i > 0; --i)
    if (SRC_PTR[i] != 0)
      break;
  for (j = mpbpl - 1; j >= 0; --j)
    if ((SRC_PTR[i] & (mp_limb_t)1 << j) != 0)
      break;

  return i * mpbpl + j;
}

static int
do_test (void)
{
  mp1 ex, x, xt, e2, e3;
  int i;
  int errors = 0;
  int failures = 0;
  mp1 maxerror;
  int maxerror_s = 0;
  const double sf = pow (2, mpbpl);

  /* assert(mpbpl == mp_bits_per_limb); */
  assert(FRAC / mpbpl * mpbpl == FRAC);

  memset (maxerror, 0, sizeof (mp1));
  memset (xt, 0, sizeof (mp1));
  xt[(FRAC - N2) / mpbpl] = (mp_limb_t)1 << (FRAC - N2) % mpbpl;

  for (i = 0; i < (1 << N2); ++i)
    {
      int e2s, e3s, j;
      double de2;

      mpn_mul_1 (x, xt, SZ, i);
      exp2_mpn (ex, x);
      de2 = exp2 (i / (double) (1 << N2));
      for (j = SZ - 1; j >= 0; --j)
	{
	  e2[j] = (mp_limb_t) de2;
	  de2 = (de2 - e2[j]) * sf;
	}
      if (mpn_cmp (ex, e2, SZ) >= 0)
	mpn_sub_n (e3, ex, e2, SZ);
      else
	mpn_sub_n (e3, e2, ex, SZ);

      e2s = mpn_bitsize (e2, SZ);
      e3s = mpn_bitsize (e3, SZ);
      if (e3s >= 0 && e2s - e3s < 54)
	{
#if PRINT_ERRORS
	  printf ("%06x ", i * (0x100000 / (1 << N2)));
	  print_mpn_fp (ex, (FRAC / 4) + 1, 16);
	  putchar ('\n');
	  fputs ("       ",stdout);
	  print_mpn_fp (e2, (FRAC / 4) + 1, 16);
	  putchar ('\n');
	  printf (" %c     ",
		  e2s - e3s < 54 ? e2s - e3s == 53 ? 'e' : 'F' : 'P');
	  print_mpn_fp (e3, (FRAC / 4) + 1, 16);
	  putchar ('\n');
#endif
	  errors += (e2s - e3s == 53);
	  failures += (e2s - e3s < 53);
	}
      if (e3s >= maxerror_s
	  && mpn_cmp (e3, maxerror, SZ) > 0)
	{
	  memcpy (maxerror, e3, sizeof (mp1));
	  maxerror_s = e3s;
	}
    }

  /* Check exp_mpn against precomputed value of exp(1).  */
  memset (x, 0, sizeof (mp1));
  x[FRAC / mpbpl] = (mp_limb_t)1 << FRAC % mpbpl;
  exp_mpn (ex, x);
  if (mpn_cmp (ex, mp_exp1, SZ) >= 0)
    mpn_sub_n (e3, ex, mp_exp1, SZ);
  else
    mpn_sub_n (e3, mp_exp1, ex, SZ);

  printf ("%d failures; %d errors; error rate %0.2f%%\n", failures, errors,
	  errors * 100.0 / (double) (1 << N2));
  fputs ("maximum error:   ", stdout);
  print_mpn_fp (maxerror, (FRAC / 4) + 1, 16);
  putchar ('\n');
  fputs ("error in exp(1): ", stdout);
  print_mpn_fp (e3, (FRAC / 4) + 1, 16);
  putchar ('\n');

  return failures == 0 ? 0 : 1;
}

#define TIMEOUT 300
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
