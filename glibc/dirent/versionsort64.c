/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#define versionsort __no_versionsort_decl
#include <dirent.h>
#undef versionsort
#include <string.h>

int
versionsort64 (const struct dirent64 **a, const struct dirent64 **b)
{
  return __strverscmp ((*a)->d_name, (*b)->d_name);
}

#if !_DIRENT_MATCHES_DIRENT64
weak_alias (versionsort64, versionsort)
#endif
