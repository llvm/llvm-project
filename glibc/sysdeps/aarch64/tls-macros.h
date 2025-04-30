/* Copyright (C) 2009-2021 Free Software Foundation, Inc.

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

#define TLS_LD(x) TLS_GD(x)

#define TLS_GD(x)					\
  ({ register unsigned long __result asm ("x0");	\
     asm ("adrp	%0, :tlsgd:" #x "; "			\
	  "add	%0, %0, #:tlsgd_lo12:" #x "; "		\
	  "bl	__tls_get_addr;"			\
	  "nop"						\
	  : "=r" (__result)				\
	  :						\
	  : "x1", "x2", "x3", "x4", "x5", "x6",		\
	    "x7", "x8", "x9", "x10", "x11", "x12",	\
	    "x13", "x14", "x15", "x16", "x17", "x18",	\
	    "x30", "memory", "cc");			\
     (int *) (__result); })

#define TLS_IE(x)					\
  ({ register unsigned long __result asm ("x0");	\
     register unsigned long __t;			\
     asm ("mrs	%1, tpidr_el0; "			\
	  "adrp	%0, :gottprel:" #x "; "			\
	  "ldr	%0, [%0, #:gottprel_lo12:" #x "]; "	\
	  "add	%0, %0, %1"				\
	  : "=r" (__result), "=r" (__t));		\
     (int *) (__result); })

#define TLS_LE(x)					\
  ({ register unsigned long __result asm ("x0");	\
     asm ("mrs	%0, tpidr_el0; "			\
	  "add	%0, %0, :tprel_hi12:" #x "; "		\
	  "add	%0, %0, :tprel_lo12_nc:" #x		\
	  : "=r" (__result));				\
     (int *) (__result); })
