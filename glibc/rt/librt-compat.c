/* Placeholder definitions to pull in removed symbol versions.
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

#if PTHREAD_IN_LIBC
# include <shlib-compat.h>
# include <sys/cdefs.h>

# if SHLIB_COMPAT (librt, GLIBC_2_1, GLIBC_2_34)
void
attribute_compat_text_section
__attribute_used__
__librt_version_placeholder_1 (void)
{
}

compat_symbol (librt, __librt_version_placeholder_1,
               __librt_version_placeholder, GLIBC_2_1);
# endif

# if SHLIB_COMPAT (librt, GLIBC_2_2, GLIBC_2_34)
compat_symbol (librt, __librt_version_placeholder_1,
               __librt_version_placeholder, GLIBC_2_2);
# endif

# if SHLIB_COMPAT (librt, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (librt, __librt_version_placeholder_1,
               __librt_version_placeholder, GLIBC_2_3_4);
# endif

# if SHLIB_COMPAT (librt, GLIBC_2_4, GLIBC_2_34)
compat_symbol (librt, __librt_version_placeholder_1,
               __librt_version_placeholder, GLIBC_2_4);
# endif

# if SHLIB_COMPAT (librt, GLIBC_2_7, GLIBC_2_34)
compat_symbol (librt, __librt_version_placeholder_1,
               __librt_version_placeholder, GLIBC_2_7);
# endif

#endif
