/* Placeholder definitions to pull in removed symbol versions.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <sys/cdefs.h>
#include <shlib-compat.h>

#ifdef SHARED
void
attribute_compat_text_section
__attribute_used__
__libpthread_version_placeholder_1 (void)
{
}
#endif

#if SHLIB_COMPAT (libpthread, GLIBC_2_0, GLIBC_2_34)
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_0);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_1, GLIBC_2_2))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_1);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_1_1, GLIBC_2_1_2))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_1_1);
#endif
#if (SHLIB_COMPAT (libpthread, GLIBC_2_1_2, GLIBC_2_2))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_1_2);
#endif

#if SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_3) \
  && ABI_libpthread_GLIBC_2_2 != ABI_libpthread_GLIBC_2_0
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_2);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_2_3, GLIBC_2_2_4))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_2_3);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_2_6, GLIBC_2_3))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_2_6);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_3_2, GLIBC_2_3_4))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_3_2);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_3_3, GLIBC_2_3_4))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_3_3);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_3_4, GLIBC_2_4))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_3_4);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_4, GLIBC_2_5))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_4);
#endif

#if SHLIB_COMPAT (libpthread, GLIBC_2_11, GLIBC_2_12)
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_11);
#endif

#if SHLIB_COMPAT (libpthread, GLIBC_2_12, GLIBC_2_13)
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_12);
#endif

#if SHLIB_COMPAT (libpthread, GLIBC_2_18, GLIBC_2_19)
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_18);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_28, GLIBC_2_29))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_28);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_30, GLIBC_2_31))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_30);
#endif

#if (SHLIB_COMPAT (libpthread, GLIBC_2_31, GLIBC_2_32))
compat_symbol (libpthread, __libpthread_version_placeholder_1,
	       __libpthread_version_placeholder, GLIBC_2_31);
#endif
