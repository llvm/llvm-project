/* Copyright (C) 2001-2021 Free Software Foundation, Inc.

   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* This file contains a bit of information about the stack allocation
   of the processor.  */

#ifndef _STACKINFO_H
# define _STACKINFO_H	1

# include <elf.h>

/* On MicroBlaze the stack grows down.  */
# define _STACK_GROWS_DOWN	1

/* Default to an executable stack.  PF_X can be overridden if PT_GNU_STACK is
 * present, but it is presumed absent.  */
# define DEFAULT_STACK_PERMS (PF_R|PF_W|PF_X)

#endif	/* stackinfo.h.  */
