/* Thread-local storage handling in the ELF dynamic linker.  IA-64 version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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


/* On IA-64 the __tls_get_addr function take the module ID and the
   offset as parameters.  */
#define GET_ADDR_ARGS		size_t tls_ia64_m, size_t tls_ia64_offset
#define GET_ADDR_PARAM		tls_ia64_m, tls_ia64_offset
#define GET_ADDR_MODULE		tls_ia64_m
#define GET_ADDR_OFFSET		tls_ia64_offset

/* We have no tls_index type.  */
#define DONT_USE_TLS_INDEX	1

extern void *__tls_get_addr (size_t m, size_t offset);
