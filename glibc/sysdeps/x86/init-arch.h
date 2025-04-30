/* This file is part of the GNU C Library.
   Copyright (C) 2008-2021 Free Software Foundation, Inc.

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

#include <ldsodefs.h>
#include <ifunc-init.h>
#include <isa.h>

#ifndef __x86_64__
/* Due to the reordering and the other nifty extensions in i686, it is
   not really good to use heavily i586 optimized code on an i686.  It's
   better to use i486 code if it isn't an i586.  */
# if MINIMUM_ISA == 686
#  define USE_I586 0
#  define USE_I686 1
# else
#  define USE_I586 (HAS_ARCH_FEATURE (I586) && !HAS_ARCH_FEATURE (I686))
#  define USE_I686 HAS_ARCH_FEATURE (I686)
# endif
#endif
