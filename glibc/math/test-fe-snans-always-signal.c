/* Test FE_SNANS_ALWAYS_SIGNAL definition.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <fenv.h>

#ifdef FE_INVALID
# ifndef FE_SNANS_ALWAYS_SIGNAL
#  ifdef __clang__
#  warning "FE_SNANS_ALWAYS_SIGNAL not defined, fix clang please"
#  else
#  error "FE_SNANS_ALWAYS_SIGNAL not defined"
#  endif
# endif
#else
# ifdef FE_SNANS_ALWAYS_SIGNAL
#  error "FE_SNANS_ALWAYS_SIGNAL defined, but no FE_INVALID support"
# endif
#endif

int
do_test (void)
{
  /* This is a compilation test.  */
  return 0;
}

#include <support/test-driver.c>
