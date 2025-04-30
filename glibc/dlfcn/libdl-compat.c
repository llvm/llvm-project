/* Placeholder compatibility symbols.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <dlfcn.h>
#include <shlib-compat.h>

/* This file is used to keep specific symbol versions occupied, so
   that ld does not generate weak symbol version definitions.  */

void
attribute_compat_text_section
__attribute_used__
__libdl_version_placeholder_1 (void)
{
}

#if SHLIB_COMPAT (libdl, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libdl, __libdl_version_placeholder_1,
               __libdl_version_placeholder, GLIBC_2_0);
#endif

#if SHLIB_COMPAT (libdl, GLIBC_2_1, GLIBC_2_34)
compat_symbol (libdl, __libdl_version_placeholder_1,
               __libdl_version_placeholder, GLIBC_2_1);
#endif

#if SHLIB_COMPAT (libdl, GLIBC_2_3_3, GLIBC_2_34)
compat_symbol (libdl, __libdl_version_placeholder_1,
               __libdl_version_placeholder, GLIBC_2_3_3);
#endif

#if SHLIB_COMPAT (libdl, GLIBC_2_3_4, GLIBC_2_34)
compat_symbol (libdl, __libdl_version_placeholder_1,
               __libdl_version_placeholder, GLIBC_2_3_4);
#endif
