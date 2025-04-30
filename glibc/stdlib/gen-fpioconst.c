/* Generate data for fpioconst.c.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <stdint.h>

int
main (void)
{
  FILE *out32 = fopen ("fpioconst-32", "w");
  if (out32 == NULL)
    abort ();
  FILE *out64 = fopen ("fpioconst-64", "w");
  if (out64 == NULL)
    abort ();
  FILE *outtable = fopen ("fpioconst-table", "w");
  if (outtable == NULL)
    abort ();
  mpz_t p;
  mpz_init (p);
  for (int i = 0; i <= 14; i++)
    {
      int j = 1 << i;
      mpz_ui_pow_ui (p, 10, j - 1);
      int exp_m = mpz_sizeinbase (p, 2);
      mpz_ui_pow_ui (p, 10, j);
      int exp_p = mpz_sizeinbase (p, 2);
      int size32 = 2 + (exp_p + 31) / 32;
      int size64 = 1 + (exp_p + 63) / 64;
      uint32_t data32[size32];
      uint64_t data64[size64];
      memset (data32, 0, sizeof data32);
      memset (data64, 0, sizeof data64);
      mpz_export (data32 + 2, NULL, -1, 4, 0, 0, p);
      mpz_export (data64 + 1, NULL, -1, 8, 0, 0, p);
      if (i == 0)
	{
	  fprintf (out32, "#define TENS_P%d_IDX\t0\n", i);
	  fprintf (out64, "#define TENS_P%d_IDX\t0\n", i);
	}
      else
	{
	  fprintf (out32, "#define TENS_P%d_IDX\t"
		   "(TENS_P%d_IDX + TENS_P%d_SIZE)\n",
		   i, i - 1, i - 1);
	  fprintf (out64, "#define TENS_P%d_IDX\t"
		   "(TENS_P%d_IDX + TENS_P%d_SIZE)\n",
		   i, i - 1, i - 1);
	}
      fprintf (out32, "#define TENS_P%d_SIZE\t%d\n", i, size32);
      fprintf (out64, "#define TENS_P%d_SIZE\t%d\n", i, size64);
      for (int k = 0; k < size32; k++)
	{
	  if (k == 0)
	    fprintf (out32, "  [TENS_P%d_IDX] = ", i);
	  else if (k % 6 == 5)
	    fprintf (out32, "\n  ");
	  else
	    fprintf (out32, " ");
	  fprintf (out32, "0x%08"PRIx32",", data32[k]);
	}
      for (int k = 0; k < size64; k++)
	{
	  if (k == 0)
	    fprintf (out64, "  [TENS_P%d_IDX] = ", i);
	  else if (k % 3 == 2)
	    fprintf (out64, "\n  ");
	  else
	    fprintf (out64, " ");
	  fprintf (out64, "0x%016"PRIx64"ull,", data64[k]);
	}
      fprintf (out32, "\n\n");
      fprintf (out64, "\n\n");
      const char *t = (i >= 10 ? "\t" : "\t\t");
      if (i == 0)
	fprintf (outtable, "  { TENS_P%d_IDX, TENS_P%d_SIZE,%s%d,\t      },\n",
		 i, i, t, exp_p);
      else
	fprintf (outtable, "  { TENS_P%d_IDX, TENS_P%d_SIZE,%s%d,\t%5d },\n",
		 i, i, t, exp_p, exp_m);
    }
  if (fclose (out32) != 0)
    abort ();
  if (fclose (out64) != 0)
    abort ();
  if (fclose (outtable) != 0)
    abort ();
  return 0;
}
