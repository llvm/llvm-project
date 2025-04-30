/* Free data allocated by a call to setlocale_r
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

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
#include <stdlib.h>
#include <libc-lock.h>

#include "localeinfo.h"


/* Lock for protecting global data.  */
__libc_rwlock_define (extern , __libc_setlocale_lock attribute_hidden)


void
__freelocale (locale_t dataset)
{
  int cnt;

  /* This static object is returned for newlocale (LC_ALL_MASK, "C").  */
  if (dataset == _nl_C_locobj_ptr)
    return;

  /* We modify global data (the usage counts).  */
  __libc_rwlock_wrlock (__libc_setlocale_lock);

  for (cnt = 0; cnt < __LC_LAST; ++cnt)
    if (cnt != LC_ALL && dataset->__locales[cnt]->usage_count != UNDELETABLE)
      /* We can remove the data.  */
      _nl_remove_locale (cnt, dataset->__locales[cnt]);

  /* It's done.  */
  __libc_rwlock_unlock (__libc_setlocale_lock);

  /* Free the locale_t handle itself.  */
  free (dataset);
}
weak_alias (__freelocale, freelocale)
