/* Builtin transformations.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

/* All encoding named must be in upper case.  There must be no extra
   spaces.  */

BUILTIN_ALIAS ("UCS4//", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("UCS-4//", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("UCS-4BE//", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("CSUCS4//", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("ISO-10646//", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("10646-1:1993//", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("10646-1:1993/UCS4/", "ISO-10646/UCS4/")
BUILTIN_ALIAS ("OSF00010104//", "ISO-10646/UCS4/") /* level 1 */
BUILTIN_ALIAS ("OSF00010105//", "ISO-10646/UCS4/") /* level 2 */
BUILTIN_ALIAS ("OSF00010106//", "ISO-10646/UCS4/") /* level 3 */

BUILTIN_TRANSFORMATION ("INTERNAL", "ISO-10646/UCS4/", 1, "=INTERNAL->ucs4",
			__gconv_transform_internal_ucs4, NULL, 4, 4, 4, 4)
BUILTIN_TRANSFORMATION ("ISO-10646/UCS4/", "INTERNAL", 1, "=ucs4->INTERNAL",
			__gconv_transform_ucs4_internal, NULL, 4, 4, 4, 4)

BUILTIN_TRANSFORMATION ("INTERNAL", "UCS-4LE//", 1, "=INTERNAL->ucs4le",
			__gconv_transform_internal_ucs4le, NULL, 4, 4, 4, 4)
BUILTIN_TRANSFORMATION ("UCS-4LE//", "INTERNAL", 1, "=ucs4le->INTERNAL",
			__gconv_transform_ucs4le_internal, NULL, 4, 4, 4, 4)

BUILTIN_ALIAS ("WCHAR_T//", "INTERNAL")

BUILTIN_ALIAS ("UTF8//", "ISO-10646/UTF8/")
BUILTIN_ALIAS ("UTF-8//", "ISO-10646/UTF8/")
BUILTIN_ALIAS ("ISO-IR-193//", "ISO-10646/UTF8/")
BUILTIN_ALIAS ("OSF05010001//", "ISO-10646/UTF8/")
BUILTIN_ALIAS ("ISO-10646/UTF-8/", "ISO-10646/UTF8/")

BUILTIN_TRANSFORMATION ("INTERNAL", "ISO-10646/UTF8/", 1, "=INTERNAL->utf8",
			__gconv_transform_internal_utf8, NULL, 4, 4, 1, 6)

BUILTIN_TRANSFORMATION ("ISO-10646/UTF8/", "INTERNAL", 1, "=utf8->INTERNAL",
			__gconv_transform_utf8_internal, __gconv_btwoc_ascii,
			1, 6, 4, 4)

BUILTIN_ALIAS ("UCS2//", "ISO-10646/UCS2/")
BUILTIN_ALIAS ("UCS-2//", "ISO-10646/UCS2/")
BUILTIN_ALIAS ("OSF00010100//", "ISO-10646/UCS2/") /* level 1 */
BUILTIN_ALIAS ("OSF00010101//", "ISO-10646/UCS2/") /* level 2 */
BUILTIN_ALIAS ("OSF00010102//", "ISO-10646/UCS2/") /* level 3 */

BUILTIN_TRANSFORMATION ("ISO-10646/UCS2/", "INTERNAL", 1, "=ucs2->INTERNAL",
			__gconv_transform_ucs2_internal, NULL, 2, 2, 4, 4)

BUILTIN_TRANSFORMATION ("INTERNAL", "ISO-10646/UCS2/", 1, "=INTERNAL->ucs2",
			__gconv_transform_internal_ucs2, NULL, 4, 4, 2, 2)


BUILTIN_ALIAS ("ANSI_X3.4//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("ISO-IR-6//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("ANSI_X3.4-1986//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("ISO_646.IRV:1991//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("ASCII//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("ISO646-US//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("US-ASCII//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("US//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("IBM367//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("CP367//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("CSASCII//", "ANSI_X3.4-1968//")
BUILTIN_ALIAS ("OSF00010020//", "ANSI_X3.4-1968//")

BUILTIN_TRANSFORMATION ("ANSI_X3.4-1968//", "INTERNAL", 1, "=ascii->INTERNAL",
			__gconv_transform_ascii_internal, __gconv_btwoc_ascii,
			1, 1, 4, 4)

BUILTIN_TRANSFORMATION ("INTERNAL", "ANSI_X3.4-1968//", 1, "=INTERNAL->ascii",
			__gconv_transform_internal_ascii, NULL, 4, 4, 1, 1)


#if BYTE_ORDER == BIG_ENDIAN
BUILTIN_ALIAS ("UNICODEBIG//", "ISO-10646/UCS2/")
BUILTIN_ALIAS ("UCS-2BE//", "ISO-10646/UCS2/")

BUILTIN_ALIAS ("UCS-2LE//", "UNICODELITTLE//")

BUILTIN_TRANSFORMATION ("UNICODELITTLE//", "INTERNAL", 1,
			"=ucs2reverse->INTERNAL",
			__gconv_transform_ucs2reverse_internal, NULL,
			2, 2, 4, 4)

BUILTIN_TRANSFORMATION ("INTERNAL", "UNICODELITTLE//", 1,
			"=INTERNAL->ucs2reverse",
			__gconv_transform_internal_ucs2reverse, NULL,
			4, 4, 2, 2)
#else
BUILTIN_ALIAS ("UNICODELITTLE//", "ISO-10646/UCS2/")
BUILTIN_ALIAS ("UCS-2LE//", "ISO-10646/UCS2/")

BUILTIN_ALIAS ("UCS-2BE//", "UNICODEBIG//")

BUILTIN_TRANSFORMATION ("UNICODEBIG//", "INTERNAL", 1,
			"=ucs2reverse->INTERNAL",
			__gconv_transform_ucs2reverse_internal, NULL,
			2, 2, 4, 4)

BUILTIN_TRANSFORMATION ("INTERNAL", "UNICODEBIG//", 1,
			"=INTERNAL->ucs2reverse",
			__gconv_transform_internal_ucs2reverse, NULL,
			4, 4, 2, 2)
#endif
