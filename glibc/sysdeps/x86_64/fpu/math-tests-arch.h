/* Runtime architecture check for math tests. x86_64 version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <sys/platform/x86.h>

#if defined REQUIRE_AVX

# define INIT_ARCH_EXT
# define CHECK_ARCH_EXT                                        \
  do                                                           \
    {                                                          \
      if (!CPU_FEATURE_ACTIVE (AVX)) return;                   \
    }                                                          \
  while (0)

#elif defined REQUIRE_AVX2

# define INIT_ARCH_EXT
# define CHECK_ARCH_EXT                                        \
  do                                                           \
    {                                                          \
      if (!CPU_FEATURE_ACTIVE (AVX2)) return;                  \
    }                                                          \
  while (0)

#elif defined REQUIRE_AVX512F

# define INIT_ARCH_EXT
# define CHECK_ARCH_EXT                                        \
  do                                                           \
    {                                                          \
      if (!CPU_FEATURE_ACTIVE (AVX512F)) return;               \
    }                                                          \
  while (0)

#else
# include <sysdeps/generic/math-tests-arch.h>
#endif
