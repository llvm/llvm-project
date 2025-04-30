/* Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#include "exit.h"


static struct exit_function_list initial_quick;
struct exit_function_list *__quick_exit_funcs = &initial_quick;

/* Register a function to be called by quick_exit.  */
int
__cxa_at_quick_exit (void (*func) (void *), void *d)
{
  return __internal_atexit (func, NULL, d, &__quick_exit_funcs);
}
