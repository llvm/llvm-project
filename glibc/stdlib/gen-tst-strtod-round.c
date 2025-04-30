/* Generate table of tests in tst-strtod-round.c from
   tst-strtod-round-data.
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

/* Compile this program as:

   gcc -std=gnu11 -O2 -Wall -Wextra gen-tst-strtod-round.c -lmpfr \
     -o gen-tst-strtod-round

   (use of current MPFR version recommended) and run it as:

   gen-tst-strtod-round tst-strtod-round-data tst-strtod-round-data.h

   The output file will be generated as tst-strtod-round-data.h
*/


#define _GNU_SOURCE
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>

/* Work around incorrect ternary value from mpfr_strtofr
   <https://sympa.inria.fr/sympa/arc/mpfr/2012-08/msg00005.html>.  */
#define WORKAROUND

static int
string_to_fp (mpfr_t f, const char *s, mpfr_rnd_t rnd)
{
  mpfr_clear_overflow ();
#ifdef WORKAROUND
  mpfr_t f2;
  mpfr_init2 (f2, 100000);
  int r0 = mpfr_strtofr (f2, s, NULL, 0, rnd);
  int r = mpfr_set (f, f2, rnd);
  r |= mpfr_subnormalize (f, r, rnd);
  mpfr_clear (f2);
  return r0 | r;
#else
  int r = mpfr_strtofr (f, s, NULL, 0, rnd);
  r |= mpfr_subnormalize (f, r, rnd);
  return r;
#endif
}

void
print_fp (FILE *fout, mpfr_t f, const char *suffix)
{
  if (mpfr_inf_p (f))
    mpfr_fprintf (fout, "\t%sINF%s", mpfr_signbit (f) ? "-" : "", suffix);
  else
    mpfr_fprintf (fout, "\t%Ra%s", f, suffix);
}

static void
round_str (FILE *fout, const char *s, int prec, int emin, int emax,
	   bool ibm_ld)
{
  mpfr_t max_value;
  mpfr_t f;
  mpfr_set_default_prec (prec);
  mpfr_set_emin (emin);
  mpfr_set_emax (emax);
  mpfr_init (f);
  int r = string_to_fp (f, s, MPFR_RNDD);
  bool overflow = mpfr_overflow_p () != 0;
  if (ibm_ld)
    {
      assert (prec == 106 && emin == -1073 && emax == 1024);
      /* The maximum value in IBM long double has discontiguous
	 mantissa bits.  */
      mpfr_init2 (max_value, 107);
      mpfr_set_str (max_value, "0x1.fffffffffffff7ffffffffffffcp+1023", 0,
		    MPFR_RNDN);
      if (mpfr_cmpabs (f, max_value) > 0)
	{
	  r = 1;
	  overflow = true;
	}
    }
  mpfr_fprintf (fout, "\t%s,\n", r ? "false" : "true");
  print_fp (fout, f, overflow ? ", true,\n" : ", false,\n");
  string_to_fp (f, s, MPFR_RNDN);
  overflow = (mpfr_overflow_p () != 0
	      || (ibm_ld && mpfr_cmpabs (f, max_value) > 0));
  print_fp (fout, f, overflow ? ", true,\n" : ", false,\n");
  string_to_fp (f, s, MPFR_RNDZ);
  overflow = (mpfr_overflow_p () != 0
	      || (ibm_ld && mpfr_cmpabs (f, max_value) > 0));
  print_fp (fout, f, overflow ? ", true,\n" : ", false,\n");
  string_to_fp (f, s, MPFR_RNDU);
  overflow = (mpfr_overflow_p () != 0
	      || (ibm_ld && mpfr_cmpabs (f, max_value) > 0));
  print_fp (fout, f, overflow ? ", true" : ", false");
  mpfr_clear (f);
  if (ibm_ld)
    mpfr_clear (max_value);
}

static void
round_for_all (FILE *fout, const char *s)
{
  static const struct fmt {
    int prec;
    int emin;
    int emax;
    bool ibm_ld;
  } formats[] = {
    { 24, -148, 128, false },
    { 53, -1073, 1024, false },
    /* This is the Intel extended float format.  */
    { 64, -16444, 16384, false },
    /* This is the Motorola extended float format.  */
    { 64, -16445, 16384, false },
    { 106, -1073, 1024, true },
    { 113, -16493, 16384, false },
  };
  mpfr_fprintf (fout, "  TEST (\"");
  const char *p;
  for (p = s; *p; p++)
    {
      fputc (*p, fout);
      if ((p - s) % 60 == 59 && p[1])
	mpfr_fprintf (fout, "\"\n\t\"");
    }
  mpfr_fprintf (fout, "\",\n");
  int i;
  int n_formats = sizeof (formats) / sizeof (formats[0]);
  for (i = 0; i < n_formats; i++)
    {
      round_str (fout, s, formats[i].prec, formats[i].emin,
		 formats[i].emax, formats[i].ibm_ld);
      if (i < n_formats - 1)
	mpfr_fprintf (fout, ",\n");
    }
  mpfr_fprintf (fout, "),\n");
}

int
main (int argc, char **argv)
{
  char *p = NULL;
  size_t len;
  ssize_t nbytes;
  FILE *fin, *fout;
  char *fin_name, *fout_name;

  if (argc < 3)
    {
      fprintf (stderr, "Usage: %s <input> <output>\n", basename (argv[0]));
      return EXIT_FAILURE;
    }

  fin_name = argv[1];
  fout_name = argv[2];

  fin = fopen (fin_name, "r");
  if (fin == NULL)
    {
      perror ("Could not open input for reading");
      return EXIT_FAILURE;
    }

  fout = fopen (fout_name, "w");
  if (fout == NULL)
    {
      perror ("Could not open output for writing");
      return EXIT_FAILURE;
    }

  fprintf (fout, "/* This file was generated by %s from %s.  */\n",
	  __FILE__, fin_name);
  fputs ("static const struct test tests[] = {\n", fout);
  while ((nbytes = getline (&p, &len, fin)) != -1)
    {
      if (p[nbytes - 1] == '\n')
	p[nbytes - 1] = 0;
      round_for_all (fout, p);
      free (p);
      p = NULL;
    }
  fputs ("};\n", fout);

  return EXIT_SUCCESS;
}
