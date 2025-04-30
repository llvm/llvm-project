/* uselocale -- fetch and set the current per-thread locale
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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

#include <locale.h>
#include "localeinfo.h"
#include <ctype.h>

/* Switch the current thread's locale to DATASET.
   If DATASET is null, instead just return the current setting.
   The special value LC_GLOBAL_LOCALE is the initial setting
   for all threads, and means the thread uses the global
   setting controlled by `setlocale'.  */
locale_t
__uselocale (locale_t newloc)
{
  locale_t oldloc = _NL_CURRENT_LOCALE;

  if (newloc != NULL)
    {
      const locale_t locobj
	= newloc == LC_GLOBAL_LOCALE ? &_nl_global_locale : newloc;
      __libc_tsd_set (locale_t, LOCALE, locobj);

#ifdef NL_CURRENT_INDIRECT
      /* Now we must update all the per-category thread-local variables to
	 point into the new current locale for this thread.  The magic
	 symbols _nl_current_LC_FOO_used are defined to meaningless values
	 if _nl_current_LC_FOO was linked in.  By using weak references to
	 both symbols and testing the address of _nl_current_LC_FOO_used,
	 we can avoid accessing the _nl_current_LC_FOO thread-local
	 variable at all when no code referring to it was linked in.  We
	 need the special bogus symbol because while TLS symbols can be
	 weak, there is no reasonable way to test for the default-zero
	 value as with a heap symbol (taking the address would just use
	 some bogus offset from our thread pointer).  */

# define DEFINE_CATEGORY(category, category_name, items, a) \
      {									      \
	extern char _nl_current_##category##_used;			      \
	weak_extern (_nl_current_##category##_used)			      \
	weak_extern (_nl_current_##category)				      \
	if (&_nl_current_##category##_used != 0)			      \
	  _nl_current_##category = &locobj->__locales[category];	      \
      }
# include "categories.def"
# undef	DEFINE_CATEGORY
#endif

      /* Update the special tsd cache of some locale data.  */
      __libc_tsd_set (const uint16_t *, CTYPE_B, (void *) locobj->__ctype_b);
      __libc_tsd_set (const int32_t *, CTYPE_TOLOWER,
		      (void *) locobj->__ctype_tolower);
      __libc_tsd_set (const int32_t *, CTYPE_TOUPPER,
		      (void *) locobj->__ctype_toupper);
    }

  return oldloc == &_nl_global_locale ? LC_GLOBAL_LOCALE : oldloc;
}
libc_hidden_def (__uselocale)
weak_alias (__uselocale, uselocale)
