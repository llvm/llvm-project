/* User interface for extracting locale-dependent parameters.
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

#include <langinfo.h>
#include <locale.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include "localeinfo.h"


/* Return a string with the data for locale-dependent parameter ITEM.  */

char *
__nl_langinfo_l (nl_item item, locale_t l)
{
  int category = _NL_ITEM_CATEGORY (item);
  unsigned int index = _NL_ITEM_INDEX (item);
  const struct __locale_data *data;

  if (category < 0 || category == LC_ALL || category >= __LC_LAST)
    /* Bogus category: bogus item.  */
    return (char *) "";

  /* Special case value for NL_LOCALE_NAME (category).
     This is not a real item index in the string table.  */
  if (index == _NL_ITEM_INDEX (_NL_LOCALE_NAME (category)))
    return (char *) l->__names[category];

#if defined NL_CURRENT_INDIRECT
  /* Make direct reference to every _nl_current_CATEGORY symbol,
     since we know only at runtime which categories are used.  */
  switch (category)
    {
# define DEFINE_CATEGORY(category, category_name, items, a) \
      case category: data = *_nl_current_##category; break;
# include "categories.def"
# undef DEFINE_CATEGORY
    default:                   /* Should be impossible.  */
      abort();
    }
#else
  data = l->__locales[category];
#endif

  if (index >= data->nstrings)
    /* Bogus index for this category: bogus item.  */
    return (char *) "";

  /* Return the string for the specified item.  */
  return (char *) data->values[index].string;
}
libc_hidden_def (__nl_langinfo_l)
weak_alias (__nl_langinfo_l, nl_langinfo_l)
