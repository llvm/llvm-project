/* Linux sys_errlist compatibility macro definitions.
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

#ifndef _ERRLIST_COMPAT_H
#define _ERRLIST_COMPAT_H

#include <shlib-compat.h>

/* Define new compat symbols for symbols  _sys_errlist, sys_errlist,
   _sys_nerr, and sys_nerr for version VERSION with NUMBERERR times number of
   bytes per long int size.
   Both _sys_errlist and sys_errlist alias to _sys_errlist_internal symbol
   (defined on errlist.c) while _sys_nerr and sys_nerr created new variable
   with the expected size.  */
#define DEFINE_COMPAT_ERRLIST(NUMBERERR, VERSION) 			     \
  const int __##VERSION##_sys_nerr = NUMBERERR;				     \
  strong_alias (__##VERSION##_sys_nerr, __##VERSION##__sys_nerr); 	     \
  declare_symbol_alias (__ ## VERSION ## _sys_errlist, _sys_errlist_internal,\
			object, NUMBERERR * (ULONG_WIDTH / UCHAR_WIDTH));    \
  declare_symbol_alias (__ ## VERSION ## __sys_errlist,			     \
			_sys_errlist_internal, object,			     \
			NUMBERERR * (ULONG_WIDTH / UCHAR_WIDTH));	     \
  compat_symbol (libc, __## VERSION ## _sys_nerr, sys_nerr, VERSION);	     \
  compat_symbol (libc, __## VERSION ## __sys_nerr, _sys_nerr, VERSION);      \
  compat_symbol (libc, __## VERSION ## _sys_errlist, sys_errlist, VERSION);  \
  compat_symbol (libc, __## VERSION ## __sys_errlist, _sys_errlist, VERSION);\

#endif
