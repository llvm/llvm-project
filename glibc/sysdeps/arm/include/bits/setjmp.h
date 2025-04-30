/* Private jmp_buf-related definitions.  ARM EABI version.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef _INCLUDE_BITS_SETJMP_H
#define _INCLUDE_BITS_SETJMP_H 1

#ifndef __ASSEMBLER__
/* Get the public declarations.  */
# include <sysdeps/arm/bits/setjmp.h>
#endif

#ifndef _ISOMAC
/* Register list for a ldm/stm instruction to load/store
   the general registers from a __jmp_buf.  */
# define JMP_BUF_REGLIST	{v1-v6, sl, fp}

/* Index of __jmp_buf where the sp register resides.  */
# define __JMP_BUF_SP		0
#endif

#endif  /* include/bits/setjmp.h */
