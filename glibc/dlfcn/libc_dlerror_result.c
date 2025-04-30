/* Thread-local variable holding the dlerror result.
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
   <http://www.gnu.org/licenses/>.  */

#include <dlerror.h>

/* This pointer is either NULL, dl_action_result_malloc_failed (), or
   has been allocated using malloc by the namespace that also contains
   this instance of the thread-local variable.  */
__thread struct dl_action_result *__libc_dlerror_result attribute_tls_model_ie;

/* Called during thread shutdown to free resources.  */
void
__libc_dlerror_result_free (void)
{
  if (__libc_dlerror_result != NULL)
    {
      if (__libc_dlerror_result != dl_action_result_malloc_failed)
        {
          dl_action_result_errstring_free (__libc_dlerror_result);
          free (__libc_dlerror_result);
        }
      __libc_dlerror_result = NULL;
    }
}
