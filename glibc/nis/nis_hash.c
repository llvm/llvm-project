/* Forward __nis_hash calls to __nss_hash, for ABI compatibility.
   Copyright (c) 2017-2021 Free Software Foundation, Inc.
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

#if SHLIB_COMPAT (libnsl, GLIBC_2_1, GLIBC_2_27)

# include <nss.h>

uint32_t
__nis_hash (const void *keyarg, size_t len)
{
  return __nss_hash (keyarg, len);
}

compat_symbol (libnsl, __nis_hash, __nis_hash, GLIBC_2_1);

#endif
