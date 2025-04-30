/* Module which depends on two modules: one NODELETE, one missing.
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

/* Note: Due to the missing second module, this object cannot be
   loaded at run time.  */

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

/* Force linking against libpthread.  */
void *pthread_create_reference = pthread_create;

/* The constructor will never be executed because the module cannot be
   loaded.  */
static void __attribute__ ((constructor))
init (void)
{
  puts ("tst-dlopenfailmod1 constructor executed");
  _exit (1);
}
