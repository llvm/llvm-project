/* Thread-local storage handling in the ELF dynamic linker.  i386 version.
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


/* Type used for the representation of TLS information in the GOT.  */
typedef struct dl_tls_index
{
  unsigned long int ti_module;
  unsigned long int ti_offset;
} tls_index;


#ifdef SHARED
/* This is the prototype for the GNU version.  */
extern void *___tls_get_addr (tls_index *ti)
     __attribute__ ((__regparm__ (1)));
extern void *___tls_get_addr_internal (tls_index *ti)
     __attribute__ ((__regparm__ (1))) attribute_hidden;

# if IS_IN (rtld)
/* The special thing about the x86 TLS ABI is that we have two
   variants of the __tls_get_addr function with different calling
   conventions.  The GNU version, which we are mostly concerned here,
   takes the parameter in a register.  The name is changed by adding
   an additional underscore at the beginning.  The Sun version uses
   the normal calling convention.  */
void *
__tls_get_addr (tls_index *ti)
{
  return ___tls_get_addr_internal (ti);
}


/* Prepare using the definition of __tls_get_addr in the generic
   version of this file.  */
# define __tls_get_addr __attribute__ ((__regparm__ (1))) ___tls_get_addr
strong_alias (___tls_get_addr, ___tls_get_addr_internal)
rtld_hidden_proto (___tls_get_addr)
rtld_hidden_def (___tls_get_addr)
#else

/* Users should get the better interface.  */
# define __tls_get_addr ___tls_get_addr

# endif
#endif
