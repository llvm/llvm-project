/* Test that variant PCS calls don't clobber registers with lazy binding.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <stdint.h>
#include <stdio.h>
#include <support/check.h>

struct regs
{
  uint64_t x[32];
  union {
    long double q[32];
    uint64_t u[64];
  } v;
};

/* Gives the registers in the caller and callee around a variant PCS call.
   Most registers are initialized from BEFORE in the caller so they can
   have values that likely show clobbers.  Register state extensions such
   as SVE is not covered here, only the base registers.  */
void vpcs_call_regs (struct regs *after, struct regs *before);

static int
do_test (void)
{
  struct regs before, after;
  int err = 0;

  unsigned char *p = (unsigned char *)&before;
  for (int i = 0; i < sizeof before; i++)
    p[i] = i & 0xff;

  vpcs_call_regs (&after, &before);

  for (int i = 0; i < 32; i++)
    if (before.x[i] != after.x[i])
      {
	if (i == 16 || i == 17)
	  /* Variant PCS allows clobbering x16 and x17.  */
	  continue;
	err++;
	printf ("x%d: before: 0x%016llx after: 0x%016llx\n",
	  i,
	  (unsigned long long)before.x[i],
	  (unsigned long long)after.x[i]);
      }
  for (int i = 0; i < 64; i++)
    if (before.v.u[i] != after.v.u[i])
      {
	err++;
	printf ("v%d: before: 0x%016llx %016llx after: 0x%016llx %016llx\n",
	  i/2,
	  (unsigned long long)before.v.u[2*(i/2)+1],
	  (unsigned long long)before.v.u[2*(i/2)],
	  (unsigned long long)after.v.u[2*(i/2)+1],
	  (unsigned long long)after.v.u[2*(i/2)]);
      }
  if (err)
    FAIL_EXIT1 ("The variant PCS call clobbered %d registers.\n", err);
  return 0;
}

#include <support/test-driver.c>
