/* Define current locale data for LC_CTYPE category.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include "localeinfo.h"
#include <ctype.h>
#include <endian.h>
#include <stdint.h>

_NL_CURRENT_DEFINE (LC_CTYPE);

/* We are called after loading LC_CTYPE data to load it into
   the variables used by the ctype.h macros.  */



void
_nl_postload_ctype (void)
{
#define current(type,x,offset) \
  ((const type *) _NL_CURRENT (LC_CTYPE, _NL_CTYPE_##x) + offset)

  const union locale_data_value *const ctypes
    = _nl_global_locale.__locales[LC_CTYPE]->values;

/* These thread-local variables are defined in ctype-info.c.
   The declarations here must match those in localeinfo.h.

   These point into arrays of 384, so they can be indexed by any `unsigned
   char' value [0,255]; by EOF (-1); or by any `signed char' value
   [-128,-1).  ISO C requires that the ctype functions work for `unsigned
   char' values and for EOF; we also support negative `signed char' values
   for broken old programs.  The case conversion arrays are of `int's
   rather than `unsigned char's because tolower (EOF) must be EOF, which
   doesn't fit into an `unsigned char'.  But today more important is that
   the arrays are also used for multi-byte character sets.

   First we update the special members of _nl_global_locale as newlocale
   would.  This is necessary for uselocale (LC_GLOBAL_LOCALE) to find these
   values properly.  */

  _nl_global_locale.__ctype_b = (const unsigned short int *)
    ctypes[_NL_ITEM_INDEX (_NL_CTYPE_CLASS)].string + 128;
  _nl_global_locale.__ctype_tolower = (const int *)
    ctypes[_NL_ITEM_INDEX (_NL_CTYPE_TOLOWER)].string + 128;
  _nl_global_locale.__ctype_toupper = (const int *)
    ctypes[_NL_ITEM_INDEX (_NL_CTYPE_TOUPPER)].string + 128;

  /* Next we must set the thread-local caches if and only if this thread is
     in fact using the global locale.  */
  if (_NL_CURRENT_LOCALE == &_nl_global_locale)
    {
      __libc_tsd_set (const uint16_t *, CTYPE_B,
		      (void *) _nl_global_locale.__ctype_b);
      __libc_tsd_set (const int32_t *, CTYPE_TOUPPER,
		      (void *) _nl_global_locale.__ctype_toupper);
      __libc_tsd_set (const int32_t *, CTYPE_TOLOWER,
		      (void *) _nl_global_locale.__ctype_tolower);
    }

#include <shlib-compat.h>
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3)
  /* We must use the exported names to access these so we are sure to
     be accessing the main executable's copy if it has COPY relocs.  */

  extern const unsigned short int *__ctype_b; /* Characteristics.  */
  extern const __int32_t *__ctype_tolower; /* Case conversions.  */
  extern const __int32_t *__ctype_toupper; /* Case conversions.  */

  extern const uint32_t *__ctype32_b;
  extern const uint32_t *__ctype32_toupper;
  extern const uint32_t *__ctype32_tolower;

  /* We need the .symver declarations these macros generate so that
     our references are explicitly bound to the versioned symbol names
     rather than the unadorned names that are not exported.  When the
     linker sees these bound to local symbols (as the unexported names are)
     then it doesn't generate a proper relocation to the global symbols.
     We need those relocations so that a versioned definition with a COPY
     reloc in an executable will override the libc.so definition.  */

compat_symbol_reference (libc, __ctype_b, __ctype_b, GLIBC_2_0);
compat_symbol_reference (libc, __ctype_tolower, __ctype_tolower, GLIBC_2_0);
compat_symbol_reference (libc, __ctype_toupper, __ctype_toupper, GLIBC_2_0);
compat_symbol_reference (libc, __ctype32_b, __ctype32_b, GLIBC_2_0);
compat_symbol_reference (libc, __ctype32_tolower, __ctype32_tolower,
			  GLIBC_2_2);
compat_symbol_reference (libc, __ctype32_toupper, __ctype32_toupper,
			 GLIBC_2_2);

  __ctype_b = current (uint16_t, CLASS, 128);
  __ctype_toupper = current (int32_t, TOUPPER, 128);
  __ctype_tolower = current (int32_t, TOLOWER, 128);
  __ctype32_b = current (uint32_t, CLASS32, 0);
  __ctype32_toupper = current (uint32_t, TOUPPER32, 0);
  __ctype32_tolower = current (uint32_t, TOLOWER32, 0);
#endif
}
