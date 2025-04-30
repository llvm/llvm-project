/* Check CPU_FEATURE_ACTIVE on IBT and SHSTK.
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

#include <x86intrin.h>
#include <sys/platform/x86.h>
#include <support/test-driver.h>
#include <support/xunistd.h>

/* Check that CPU_FEATURE_ACTIVE on IBT and SHSTK matches _get_ssp.  */

static int
do_test (void)
{
  if (_get_ssp () != 0)
    {
      if (CPU_FEATURE_ACTIVE (IBT) && CPU_FEATURE_ACTIVE (SHSTK))
	return EXIT_SUCCESS;
    }
  else
    {
      if (!CPU_FEATURE_ACTIVE (IBT) && !CPU_FEATURE_ACTIVE (SHSTK))
	return EXIT_SUCCESS;
    }

  return EXIT_FAILURE;
}

#include <support/test-driver.c>
