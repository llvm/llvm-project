/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/profil.h>

#include <bits/wordsize.h>

#define NELEMS(a)	(sizeof (a)/sizeof ((a)[0]))

size_t taddr[] =
  {
    0x00001000,		/* elf32/hppa */
    0x08048000,		/* Linux elf32/x86 */
    0x80000000,		/* Linux elf32/m68k */
    0x00400000,		/* Linux elf32/mips */
    0x01800000,		/* Linux elf32/ppc */
    0x00010000		/* Linux elf32/sparc */
#if __WORDSIZE > 32
    ,
    0x4000000000000000,	/* Linux elf64/ia64 */
    0x0000000120000000,	/* Linux elf64/alpha */
    0x4000000000001000,	/* elf64/hppa */
    0x0000000100000000	/* Linux elf64/sparc */
#endif
  };

unsigned int buf[NELEMS (taddr)][0x10000 / sizeof (int)];
unsigned int bshort[5][0x100 / sizeof (int)];
unsigned int blong[1][0x1000 / sizeof (int)];
unsigned int vlong[1][0x2000 / sizeof (int)];

static long int
fac (long int n)
{
  if (n == 0)
    return 1;
  return n * fac (n - 1);
}

int
main (int argc, char **argv)
{
  unsigned int ovfl = 0, profcnt = 0;
  struct timeval tv, start;
  struct prof prof[32];
  double t_tick, delta;
  long int sum = 0;
  int i, j;

  for (i = 0; i < NELEMS (taddr); ++i)
    {
      prof[profcnt].pr_base = buf[i];
      prof[profcnt].pr_size = sizeof (buf[i]);
      prof[profcnt].pr_off = taddr[i];
      prof[profcnt].pr_scale = 0x10000;
      ++profcnt;
    }

  prof[profcnt].pr_base = blong[0];
  prof[profcnt].pr_size = sizeof (blong[0]);
  prof[profcnt].pr_off = 0x80001000;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  prof[profcnt].pr_base = bshort[0];
  prof[profcnt].pr_size = sizeof (bshort[0]);
  prof[profcnt].pr_off = 0x80000080;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  prof[profcnt].pr_base = bshort[1];
  prof[profcnt].pr_size = sizeof (bshort[1]);
  prof[profcnt].pr_off = 0x80000f80;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  prof[profcnt].pr_base = bshort[2];
  prof[profcnt].pr_size = sizeof (bshort[2]);
  prof[profcnt].pr_off = 0x80001080;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  prof[profcnt].pr_base = bshort[3];
  prof[profcnt].pr_size = sizeof (bshort[3]);
  prof[profcnt].pr_off = 0x80001f80;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  prof[profcnt].pr_base = bshort[4];
  prof[profcnt].pr_size = sizeof (bshort[4]);
  prof[profcnt].pr_off = 0x80002080;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  prof[profcnt].pr_base = vlong[0];
  prof[profcnt].pr_size = sizeof (vlong[0]);
  prof[profcnt].pr_off = 0x80000080;
  prof[profcnt].pr_scale = 0x10000;
  ++profcnt;

  /* Set up overflow counter (must be last on Irix).  */
  prof[profcnt].pr_base = &ovfl;
  prof[profcnt].pr_size = sizeof (ovfl);
  prof[profcnt].pr_off = 0;
  prof[profcnt].pr_scale = 2;
  ++profcnt;

  /* Turn it on.  */
  if (sprofil (prof, profcnt, &tv, PROF_UINT) < 0)
    {
      if (errno == ENOSYS)
	exit (0);
      perror ("sprofil");
      exit (1);
    }

  t_tick = tv.tv_sec + 1e-6 * tv.tv_usec;
  printf ("profiling period = %g ms\n", 1e3 * t_tick);

  gettimeofday (&start, NULL);
  do
    {
      for (i = 0; i < 21; ++i)
	sum += fac (i);

      gettimeofday (&tv, NULL);
      timersub (&tv, &start, &tv);
      delta = tv.tv_sec + 1e-6 * tv.tv_usec;
    }
  while (delta < 1000 * t_tick);

  printf ("sum = 0x%lx\n", sum);

  /* Turn it off.  */
  if (sprofil (NULL, 0, NULL, 0) < 0)
    {
      if (errno == ENOSYS)
	exit (0);
      perror ("sprofil");
      exit (1);
    }

  printf ("overflow = %u\n", ovfl);
  for (i = 0; i < NELEMS (taddr); ++i)
    for (j = 0; j < 0x10000 / sizeof (int); ++j)
      if (buf[i][j] != 0)
	printf ("%0*Zx\t%u\t(buffer %d)\n",
		(int) (sizeof (size_t) * 2),
		(taddr[i] + ((char *) &buf[i][j] - (char *) &buf[i][0])),
		buf[i][j], i);

  return 0;
}
