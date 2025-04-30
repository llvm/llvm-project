/* Common definition for memcpy resolver.
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

#ifdef __SOFTFP__
__typeof (REDIRECT_NAME) OPTIMIZE (arm) attribute_hidden;
#endif
__typeof (REDIRECT_NAME) OPTIMIZE (vfp) attribute_hidden;
__typeof (REDIRECT_NAME) OPTIMIZE (neon) attribute_hidden;

static inline void *
IFUNC_SELECTOR (int hwcap)
{
  if (hwcap & HWCAP_ARM_NEON)
    return OPTIMIZE (neon);
#ifdef __SOFTFP__
  if (hwcap & HWCAP_ARM_VFP)
    return OPTIMIZE (vfp);
  return OPTIMIZE (arm);
#else
  return OPTIMIZE (vfp);
#endif
}
