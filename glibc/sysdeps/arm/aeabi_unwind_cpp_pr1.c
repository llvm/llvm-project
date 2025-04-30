/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Because some objects in ld.so and libc.so are built with
   -fexceptions, we end up with references to this personality
   routine.  However, these libraries are not linked against
   libgcc_eh.a, so we need a dummy definition.   This routine will
   never actually be called.  */

#include <stdlib.h>

attribute_hidden
void
__aeabi_unwind_cpp_pr0 (void)
{
#if !IS_IN (rtld)
  abort ();
#endif
}

attribute_hidden
void
__aeabi_unwind_cpp_pr1 (void)
{
#if !IS_IN (rtld)
  abort ();
#endif
}

attribute_hidden
void
__aeabi_unwind_cpp_pr2 (void)
{
#if !IS_IN (rtld)
  abort ();
#endif
}
