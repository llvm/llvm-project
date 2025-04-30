/* Check ISA level on dlopened shared object.
   Copyright (C) 2020 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <elf.h>
#include <get-isa-level.h>
#include <support/xdlfcn.h>
#include <support/check.h>
#include <support/test-driver.h>

static void
do_test_1 (const char *modname, bool fail)
{
  int (*fp) (void);
  void *h;

  h = dlopen (modname, RTLD_LAZY);
  if (h == NULL)
    {
      const char *err = dlerror ();
      if (fail)
	{
	  if (strstr (err, "CPU ISA level is lower than required") == NULL)
	    FAIL_EXIT1 ("incorrect dlopen '%s' error: %s\n", modname, err);

	  return;
	}

      FAIL_EXIT1 ("cannot open '%s': %s\n", modname, err);
    }

  if (fail)
    FAIL_EXIT1 ("dlopen '%s' should have failed\n", modname);

  fp = xdlsym (h, "test");

  if (fp () != 0)
    FAIL_EXIT1 ("test () != 0\n");

  dlclose (h);
}

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
    {
      do_test_1 ("tst-isa-level-mod-1-baseline.so", true);
      return EXIT_SUCCESS;
    }

  do_test_1 ("tst-isa-level-mod-1-baseline.so", false);

  /* Skip on x86-64-v4 platforms since dlopen v4 module always works.  */
  if (has_isa_v4)
    return EXIT_SUCCESS;

  do_test_1 ("tst-isa-level-mod-1-v4.so", true);

  /* Skip on x86-64-v3 platforms since dlopen v3 module always works.  */
  if (has_isa_v3)
    return EXIT_SUCCESS;

  do_test_1 ("tst-isa-level-mod-1-v3.so", true);

  /* Skip on x86-64-v2 platforms since dlopen v2 module always works.  */
  if (has_isa_v2)
    return EXIT_SUCCESS;

  do_test_1 ("tst-isa-level-mod-1-v2.so", true);

  return EXIT_SUCCESS;
}

#include <support/test-driver.c>
