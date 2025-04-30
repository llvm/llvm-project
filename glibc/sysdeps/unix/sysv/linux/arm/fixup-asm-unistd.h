/* Regularize <asm/unistd.h> definitions.  Arm version.
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

/* These system calls have iregular names and are used by glibc.  */
#ifndef __NR_cacheflush
# define __NR_cacheflush __ARM_NR_cacheflush
#endif
#ifndef __NR_set_tls
# define __NR_set_tls    __ARM_NR_set_tls
#endif
