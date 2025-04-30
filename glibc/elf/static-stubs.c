/* Stub implementations of functions to link into statically linked
   programs without needing libgcc_eh.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Avoid backtrace (and so _Unwind_Backtrace) dependencies from
   sysdeps/unix/sysv/linux/libc_fatal.c.  */
#include <sysdeps/posix/libc_fatal.c>

#include <stdlib.h>
#include <unwind.h>

/* These programs do not use thread cancellation, so _Unwind_Resume
   and the personality routine are never actually called.  */

void
_Unwind_Resume (struct _Unwind_Exception *exc __attribute__ ((unused)))
{
  abort ();
}

_Unwind_Reason_Code
__gcc_personality_v0 (int version __attribute__ ((unused)),
		      _Unwind_Action actions __attribute__ ((unused)),
		      _Unwind_Exception_Class exception_class
		      __attribute__ ((unused)),
		      struct _Unwind_Exception *ue_header
		      __attribute__ ((unused)),
		      struct _Unwind_Context *context __attribute__ ((unused)))
{
  abort ();
}
