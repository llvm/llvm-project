/* Regularize <asm/unistd.h> definitions.  Default version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

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

/* Some architectures have irregular system call names in
   <asm/unistd.h>.  glibc assumes that system call numbers start with
   __NR_* and lists the system calls under proper names in
   <arch-syscall.h>.

   During consistency tests, <fixup-asm-unistd.h> is included after
   the kernel's <asm/unistd.h>, to introduce aliases as necessary to
   match the glibc definitions in <arch-syscall.h>.

   Most architectures do not need these fixups, so the default header
   is empty.  */
