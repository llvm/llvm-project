/* Install given context.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <sysdep.h>
#include <ucontext-offsets.h>


/* In case the user fiddled it, copy the "official" signal mask
   from the ucontext_t into the sigcontext structure.  */
#undef PSEUDO_PREPARE_ARGS
#define PSEUDO_PREPARE_ARGS			\
	ldq	$0, UC_SIGMASK($16);		\
	stq	$0, UC_SIGCTX+SC_MASK($16);	\
	lda	$16, UC_SIGCTX($16);

PSEUDO(__setcontext, sigreturn, 1)
	ret
PSEUDO_END(__setcontext)
weak_alias (__setcontext, setcontext)
