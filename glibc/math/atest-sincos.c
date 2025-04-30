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

#define PRINT_ERRORS 0

#define N 0
#define N2 20
#define FRAC (32 * 4)

#define mpbpl (CHAR_BIT * sizeof (mp_limb_t))
#define SZ (FRAC / mpbpl + 1)
typedef mp_limb_t mp1[SZ], mp2[SZ * 2];

/* These strings have exactly 100 hex digits in them.  */
static const char sin1[101] =
"d76aa47848677020c6e9e909c50f3c3289e511132f518b4def"
"b6ca5fd6c649bdfb0bd9ff1edcd4577655b5826a3d3b50c264";
static const char cos1[101] =
"8a51407da8345c91c2466d976871bd29a2373a894f96c3b7f2"
"300240b760e6fa96a94430a52d0e9e43f3450e3b8ff99bc934";
static const char hexdig[] = "0123456789abcdef";

static void
print_mpn_hex (const mp_limb_t *x, unsigned size)
{
   char value[size + 1];
   unsigned i;
   const unsigned final = (size * 4 > SZ * mpbpl) ? SZ * mpbpl / 4 : size;

   memset (value, '0', size);

   for (i = 0; i < final ; i++)
     value[size-1-i] = hexdig[x[i * 4 / mpbpl] >> (i * 4) % mpbpl & 0xf];

   value[size] = '\0';
   fputs (value, stdout);
}

static void
sincosx_mpn (mp1 si, mp1 co, mp1 xx, mp1 ix)
{
   int i;
   mp2 s[4], c[4];
   mp1 tmp, x;

   if (ix == NULL)
     {
       memset (si, 0, sizeof (mp1));
       memset (co, 0, sizeof (mp1));
       co[SZ-1] = 1;
       memcpy (x, xx, sizeof (mp1));
     }
   else
      mpn_sub_n (x, xx, ix, SZ);

   for (i = 0; i < 1 << N; i++)
     {
#define add_shift_mulh(d,x,s1,s2,sh,n) \
      do { 								      \
	 if (s2 != NULL) {						      \
	    if (sh > 0) {						      \
	       assert (sh < mpbpl);					      \
	       mpn_lshift (tmp, s1, SZ, sh);				      \
	       if (n)							      \
	         mpn_sub_n (tmp,tmp,s2+FRAC/mpbpl,SZ);			      \
	       else							      \
	         mpn_add_n (tmp,tmp,s2+FRAC/mpbpl,SZ);			      \
	    } else {							      \
	       if (n)							      \
	         mpn_sub_n (tmp,s1,s2+FRAC/mpbpl,SZ);			      \
	       else							      \
	         mpn_add_n (tmp,s1,s2+FRAC/mpbpl,SZ);			      \
	    }								      \
	    mpn_mul_n(d,tmp,x,SZ);					      \
	 } else 							      \
	    mpn_mul_n(d,s1,x,SZ);					      \
	 assert(N+sh < mpbpl);						      \
	 if (N+sh > 0) mpn_rshift(d,d,2*SZ,N+sh);			      \
      } while(0)
#define summ(d,ss,s,n) \
      do { 								      \
	 mpn_add_n(tmp,s[1]+FRAC/mpbpl,s[2]+FRAC/mpbpl,SZ);		      \
	 mpn_lshift(tmp,tmp,SZ,1);					      \
	 mpn_add_n(tmp,tmp,s[0]+FRAC/mpbpl,SZ);				      \
	 mpn_add_n(tmp,tmp,s[3]+FRAC/mpbpl,SZ);				      \
	 mpn_divmod_1(tmp,tmp,SZ,6);					      \
	 if (n)								      \
           mpn_sub_n (d,ss,tmp,SZ);					      \
	 else								      \
           mpn_add_n (d,ss,tmp,SZ);					      \
      } while (0)

      add_shift_mulh (s[0], x, co, NULL, 0, 0); /* s0 = h * c; */
      add_shift_mulh (c[0], x, si, NULL, 0, 0); /* c0 = h * s; */
      add_shift_mulh (s[1], x, co, c[0], 1, 1); /* s1 = h * (c - c0/2); */
      add_shift_mulh (c[1], x, si, s[0], 1, 0); /* c1 = h * (s + s0/2); */
      add_shift_mulh (s[2], x, co, c[1], 1, 1); /* s2 = h * (c - c1/2); */
      add_shift_mulh (c[2], x, si, s[1], 1, 0); /* c2 = h * (s + s1/2); */
      add_shift_mulh (s[3], x, co, c[2], 0, 1); /* s3 = h * (c - c2); */
      add_shift_mulh (c[3], x, si, s[2], 0, 0); /* c3 = h * (s + s2); */
      summ (si, si, s, 0);        /* s = s + (s0+2*s1+2*s2+s3)/6; */
      summ (co, co, c, 1);        /* c = c - (c0+2*c1+2*c2+c3)/6; */
   }
#undef add_shift_mulh
#undef summ
}

static int
mpn_bitsize (const mp_limb_t *SRC_PTR, mp_size_t SIZE)
{
   int i, j;
   for (i = SIZE - 1; i > 0; i--)
     if (SRC_PTR[i] != 0)
       break;
   for (j = mpbpl - 1; j >= 0; j--)
     if ((SRC_PTR[i] & (mp_limb_t)1 << j) != 0)
       break;

   return i * mpbpl + j;
}

static int
do_test (void)
{
  mp1 si, co, x, ox, xt, s2, c2, s3, c3;
  int i;
  int sin_errors = 0, cos_errors = 0;
  int sin_failures = 0, cos_failures = 0;
  mp1 sin_maxerror, cos_maxerror;
  int sin_maxerror_s = 0, cos_maxerror_s = 0;
  const double sf = pow (2, mpbpl);

  /* assert(mpbpl == mp_bits_per_limb);  */
  assert(FRAC / mpbpl * mpbpl == FRAC);

  memset (sin_maxerror, 0, sizeof (mp1));
  memset (cos_maxerror, 0, sizeof (mp1));
  memset (xt, 0, sizeof (mp1));
  xt[(FRAC - N2) / mpbpl] = (mp_limb_t)1 << (FRAC - N2) % mpbpl;

  for (i = 0; i < 1 << N2; i++)
    {
      int s2s, s3s, c2s, c3s, j;
      double ds2,dc2;

      mpn_mul_1 (x, xt, SZ, i);
      sincosx_mpn (si, co, x, i == 0 ? NULL : ox);
      memcpy (ox, x, sizeof (mp1));
      ds2 = sin (i / (double) (1 << N2));
      dc2 = cos (i / (double) (1 << N2));
      for (j = SZ-1; j >= 0; j--)
	{
	  s2[j] = (mp_limb_t) ds2;
	  ds2 = (ds2 - s2[j]) * sf;
	  c2[j] = (mp_limb_t) dc2;
	  dc2 = (dc2 - c2[j]) * sf;
	}
      if (mpn_cmp (si, s2, SZ) >= 0)
	mpn_sub_n (s3, si, s2, SZ);
      else
	mpn_sub_n (s3, s2, si, SZ);
      if (mpn_cmp (co, c2, SZ) >= 0)
	mpn_sub_n (c3, co, c2, SZ);
      else
	mpn_sub_n (c3, c2, co, SZ);

      s2s = mpn_bitsize (s2, SZ);
      s3s = mpn_bitsize (s3, SZ);
      c2s = mpn_bitsize (c2, SZ);
      c3s = mpn_bitsize (c3, SZ);
      if ((s3s >= 0 && s2s - s3s < 54)
	  || (c3s >= 0 && c2s - c3s < 54)
	  || 0)
	{
#if PRINT_ERRORS
	  printf ("%06x ", i * (0x100000 / (1 << N2)));
	  print_mpn_hex(si, (FRAC / 4) + 1);
	  putchar (' ');
	  print_mpn_hex (co, (FRAC / 4) + 1);
	  putchar ('\n');
	  fputs ("       ", stdout);
	  print_mpn_hex (s2, (FRAC / 4) + 1);
	  putchar (' ');
	  print_mpn_hex (c2, (FRAC / 4) + 1);
	  putchar ('\n');
	  printf (" %c%c    ",
		  s3s >= 0 && s2s-s3s < 54 ? s2s - s3s == 53 ? 'e' : 'F' : 'P',
		  c3s >= 0 && c2s-c3s < 54 ? c2s - c3s == 53 ? 'e' : 'F' : 'P');
	  print_mpn_hex (s3, (FRAC / 4) + 1);
	  putchar (' ');
	  print_mpn_hex (c3, (FRAC / 4) + 1);
	  putchar ('\n');
#endif
	  sin_errors += s2s - s3s == 53;
	  cos_errors += c2s - c3s == 53;
	  sin_failures += s2s - s3s < 53;
	  cos_failures += c2s - c3s < 53;
	}
      if (s3s >= sin_maxerror_s
	  && mpn_cmp (s3, sin_maxerror, SZ) > 0)
	{
	  memcpy (sin_maxerror, s3, sizeof (mp1));
	  sin_maxerror_s = s3s;
	}
      if (c3s >= cos_maxerror_s
	  && mpn_cmp (c3, cos_maxerror, SZ) > 0)
	{
	  memcpy (cos_maxerror, c3, sizeof (mp1));
	  cos_maxerror_s = c3s;
	}
    }

   /* Check Range-Kutta against precomputed values of sin(1) and cos(1).  */
   memset (x, 0, sizeof (mp1));
   x[FRAC / mpbpl] = (mp_limb_t)1 << FRAC % mpbpl;
   sincosx_mpn (si, co, x, ox);

   memset (s2, 0, sizeof (mp1));
   memset (c2, 0, sizeof (mp1));
   for (i = 0; i < 100 && i < FRAC / 4; i++)
     {
       s2[(FRAC - i * 4 - 4) / mpbpl] |= ((mp_limb_t) (strchr (hexdig, sin1[i])
						       - hexdig)
 					  << (FRAC - i * 4 - 4) % mpbpl);
       c2[(FRAC - i * 4 - 4) / mpbpl] |= ((mp_limb_t) (strchr (hexdig, cos1[i])
						       - hexdig)
					  << (FRAC - i * 4 - 4) % mpbpl);
     }

   if (mpn_cmp (si, s2, SZ) >= 0)
     mpn_sub_n (s3, si, s2, SZ);
   else
     mpn_sub_n (s3, s2, si, SZ);
   if (mpn_cmp (co, c2, SZ) >= 0)
      mpn_sub_n (c3, co, c2, SZ);
   else
     mpn_sub_n (c3, c2, co, SZ);

   printf ("sin:\n");
   printf ("%d failures; %d errors; error rate %0.2f%%\n",
	   sin_failures, sin_errors, sin_errors * 100.0 / (double) (1 << N2));
   fputs ("maximum error:   ", stdout);
   print_mpn_hex (sin_maxerror, (FRAC / 4) + 1);
   fputs ("\nerror in sin(1): ", stdout);
   print_mpn_hex (s3, (FRAC / 4) + 1);

   fputs ("\n\ncos:\n", stdout);
   printf ("%d failures; %d errors; error rate %0.2f%%\n",
	   cos_failures, cos_errors, cos_errors * 100.0 / (double) (1 << N2));
   fputs ("maximum error:   ", stdout);
   print_mpn_hex (cos_maxerror, (FRAC / 4) + 1);
   fputs ("\nerror in cos(1): ", stdout);
   print_mpn_hex (c3, (FRAC / 4) + 1);
   putchar ('\n');

   return (sin_failures == 0 && cos_failures == 0) ? 0 : 1;
}

#define TIMEOUT 600
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
