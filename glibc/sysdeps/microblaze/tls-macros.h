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

#define TLS_LD(x)                                  \
  ({                                               \
    char *__result;                                \
    int __offset;                                  \
    extern void *__tls_get_addr (void *);          \
    asm ("mfs r20,rpc \n"                          \
         "addik r20,r20,_GLOBAL_OFFSET_TABLE_+8\n" \
         "addik %0,r20," #x "@TLSLDM"              \
         : "=r" (__result));                       \
    __result = (char *) __tls_get_addr (__result); \
    asm ("addik %0,r0,"#x"@TLSDTPREL"              \
         : "=r" (__offset));                       \
    (int *) (__result + __offset); })


#define TLS_GD(x)                                  \
  ({                                               \
    int *__result;                                 \
    extern void *__tls_get_addr (void *);          \
    asm ("mfs  r20,rpc\n"                          \
         "addik r20,r20,_GLOBAL_OFFSET_TABLE_+8\n" \
         "addik %0,r20," #x "@TLSGD"               \
         : "=r" (__result));                       \
    (int *) __tls_get_addr (__result); })

#define TLS_LE(x) TLS_LD(x)

#define TLS_IE(x) TLS_GD(x)
