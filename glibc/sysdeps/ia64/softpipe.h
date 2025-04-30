/* This file is part of the GNU C Library.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.

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

/* The latency of a memory load assumed by the assembly implementation
   of the mem and str functions.  Since we don't have any clue about
   where the data might be, let's assume it's in the L2 cache.
   Assuming L3 would be too pessimistic :-)

   Some functions define MEMLAT as 2, because they expect their data
   to be in the L1D cache.  */

#ifndef MEMLAT
# define MEMLAT 6
#endif
