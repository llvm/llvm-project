/* DSO which links against libpthread and triggers a lazy binding.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* This file is compiled into a DSO which loads libpthread, but fails
   the dynamic linker afterwards.  */

#include <pthread.h>

/* Link in libpthread.  */
void *pthread_create_ptr = &pthread_create;

int this_function_is_not_defined (void);

int
trigger_dynlink_failure (void)
{
  return this_function_is_not_defined ();
}
