/* Macros to support TLS testing in times of missing compiler support.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.

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


#include <sys/cdefs.h>
#include <sys/asm.h>
#include <sysdep.h>
#include "dl-tls.h"

#define TLS_GD(x)					\
	({ void *__result;				\
	asm ("la.tls.gd %0, " #x "\n\t"			\
	     : "=r" (__result));			\
	__tls_get_addr (__result); })

#define TLS_LD(x) TLS_GD(x)

#define TLS_IE(x)					\
	({ void *__result;				\
	asm ("la.tls.ie %0, " #x "\n\t"			\
	     "add %0, %0, tp\n\t"			\
	     : "=r" (__result));			\
	__result; })

#define TLS_LE(x)					\
	({ void *__result;				\
	asm ("lui %0, %%tprel_hi(" #x ")\n\t"		\
	     "add %0, %0, tp, %%tprel_add(" #x ")\n\t"	\
	     "addi %0, %0, %%tprel_lo(" #x ")\n\t"	\
	     : "=r" (__result));			\
	__result; })
