/* Verify that DSO is unloaded only if its TLS objects are destroyed - the DSO.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <dso_handle.h>

typedef struct
{
  void *val;
} A;

/* We only care about the destructor.  */
void A_dtor (void *obj)
{
  ((A *)obj)->val = obj;
}

void reg_dtor (void)
{
  static __thread A b;
  __cxa_thread_atexit_impl (A_dtor, &b, __dso_handle);
}
