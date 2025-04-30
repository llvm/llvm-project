/* Macros for accessing thread-local storage.  C-SKY ABIV2 version.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

# define TLS_LE(x)					\
  ({ int *__result;					\
     __asm__ ("lrw %0, " #x "@TPOFF\n\t"		\
	      "add %0, tls, %0"				\
	      : "=&r" (__result));			\
     __result; })

# define TLS_IE(x)					\
  ({ int *__result;					\
     __asm__ ("grs a1, 1f\n"				\
	      "1:\tlrw %0, " #x "@GOTTPOFF\n\t"		\
	      "ldr.w %0, (a1, %0 << 0)\n\t"		\
	      "add %0, tls, %0"				\
	      : "=&r" (__result): : "a1");		\
     __result; })

# define TLS_LD(x)					\
  ({ char *__result;					\
     int __offset;					\
     extern void *__tls_get_addr (void *);		\
     __asm__ ("grs a1, 1f\n"				\
	      "1:\tlrw %0, " #x "@TLSLDM32;\n\t"	\
	      "add %0, a1, %0"				\
	      : "=r" (__result) : : "a1");		\
     __result = (char *)__tls_get_addr (__result);	\
     __asm__ ("lrw %0, " #x "@TLSLDO32"			\
	      : "=r" (__offset));			\
     (int *) (__result + __offset); })

# define TLS_GD(x)					\
  ({ int *__result;					\
     extern void *__tls_get_addr (void *);		\
     __asm__ ("grs a1, 1f\n"				\
	      "1:\tlrw %0, " #x "@TLSGD32\n\t"		\
	      "add %0, a1, %0"				\
	      : "=r" (__result) : : "a1");		\
     (int *)__tls_get_addr (__result); })
