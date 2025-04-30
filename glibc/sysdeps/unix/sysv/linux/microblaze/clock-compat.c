/* ABI compatibility redirects for clock_* symbols in librt.
   MicroBlaze version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>

/* These symbols were accidentally included in librt for MicroBlaze
   despite the first release coming after the general obsoletion in
   librt, so ensure they remain as part of the ABI.  */

#ifdef SHARED
# undef SHLIB_COMPAT
# define SHLIB_COMPAT(lib, introduced, obsoleted) 1
#endif

#include <rt/clock-compat.c>
