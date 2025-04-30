/* Check ISA level on shared object in glibc-hwcaps subdirectories.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <elf.h>
#include <get-isa-level.h>
#include <support/check.h>
#include <support/test-driver.h>

extern int dso_isa_level (void);

static int
do_test (void)
{
  const struct cpu_features *cpu_features = __get_cpu_features ();
  unsigned int isa_level = get_isa_level (cpu_features);
  bool has_isa_baseline = ((isa_level & GNU_PROPERTY_X86_ISA_1_BASELINE)
			   == GNU_PROPERTY_X86_ISA_1_BASELINE);
  bool has_isa_v2 = ((isa_level & GNU_PROPERTY_X86_ISA_1_V2)
			   == GNU_PROPERTY_X86_ISA_1_V2);
  bool has_isa_v3 = ((isa_level & GNU_PROPERTY_X86_ISA_1_V3)
			   == GNU_PROPERTY_X86_ISA_1_V3);
  bool has_isa_v4 = ((isa_level & GNU_PROPERTY_X86_ISA_1_V4)
			   == GNU_PROPERTY_X86_ISA_1_V4);

  if (!has_isa_baseline)
    return EXIT_FAILURE;

  int level = dso_isa_level ();
  int ret;
  switch (level)
    {
    case 1:
    case 2:
      /* The default libx86-64-isa-level.so is used.  */
      printf ("The default shared library is used.\n");
      if (has_isa_v3 || has_isa_v4 || (!has_isa_v2 && level == 2))
	ret = EXIT_FAILURE;
      else
	ret = EXIT_SUCCESS;
      break;
    case 3:
      /* libx86-64-isa-level.so marked as x86-64 ISA level 3 needed in
	 x86-64-v2 should be ignored on lesser CPU.  */
      printf ("x86-64 ISA level 3 shared library is used.\n");
      if (has_isa_v4 || !has_isa_v3)
	ret = EXIT_FAILURE;
      else
	ret = EXIT_SUCCESS;
      break;
    case 4:
      /* libx86-64-isa-level.so marked as x86-64 ISA level 4 needed in
	 x86-64-v3 should be ignored on lesser CPU.  */
      printf ("x86-64 ISA level 4 shared library is used.\n");
      if (has_isa_v4)
	ret = EXIT_SUCCESS;
      else
	ret = EXIT_FAILURE;
      break;
    default:
      abort ();
    }
  return ret;
}

#include <support/test-driver.c>
