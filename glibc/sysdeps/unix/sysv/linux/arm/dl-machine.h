/* Machine-dependent ELF dynamic relocation inline functions.  ARM/Linux version
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef dl_machine_h

/* This definition is Linux-specific.  */
#define CLEAR_CACHE(BEG,END)                                            \
  INTERNAL_SYSCALL_CALL (cacheflush, (BEG), (END), 0)

#endif

/* The rest is just machine-specific.
   This #include is outside the #ifndef because the parts of
   dl-machine.h used only by dynamic-link.h are outside the guard.  */
#include <sysdeps/arm/dl-machine.h>
