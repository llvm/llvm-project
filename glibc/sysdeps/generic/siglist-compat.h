/* Generic siglist compatibility macro definitions.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef _SIGLIST_COMPAT_H
#define _SIGLIST_COMPAT_H

#include <shlib-compat.h>
#include <limits.h>

/* Define new compat symbols for sys_siglist, _sys_siglist, and sys_sigabbrev
   for version VERSION with NUMBERSIG times the number of bytes per long int.
   Both _sys_siglist and sys_siglist alias to __sys_siglist while
   sys_sigabbrev alias to __sys_sigabbrev.  Both target alias are
   define in siglist.c.  */
#define DEFINE_COMPAT_SIGLIST(NUMBERSIG, VERSION) 			     \
  declare_symbol_alias (__ ## VERSION ## _sys_siglist,			     \
			__sys_siglist,					     \
			object,	NUMBERSIG * (ULONG_WIDTH / UCHAR_WIDTH));    \
  declare_symbol_alias (__ ## VERSION ## sys_siglist,			     \
			__sys_siglist,					     \
			object,	NUMBERSIG * (ULONG_WIDTH / UCHAR_WIDTH));    \
  declare_symbol_alias (__ ## VERSION ## _sys_sigabbrev,		     \
			__sys_sigabbrev,				     \
			object, NUMBERSIG * (ULONG_WIDTH / UCHAR_WIDTH));    \
  compat_symbol (libc, __## VERSION ## _sys_siglist,   _sys_siglist,	     \
		 VERSION);						     \
  compat_symbol (libc, __## VERSION ## sys_siglist,    sys_siglist,	     \
		 VERSION);						     \
  compat_symbol (libc, __## VERSION ## _sys_sigabbrev, sys_sigabbrev,	     \
		 VERSION);						     \

#endif
