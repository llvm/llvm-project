/* Thread-local storage handling in the ELF dynamic linker.  s390 version.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
typedef struct
{
  unsigned long int ti_module;
  unsigned long int ti_offset;
} tls_index;


#ifdef SHARED

extern unsigned long __tls_get_offset (unsigned long got_offset);

# if IS_IN (rtld)

#  include <shlib-compat.h>

/* dl-tls.c declares __tls_get_addr as an exported symbol if it is not defined
   as a macro.  It seems suitable to do that in the generic code because all
   architectures other than s390 export __tls_get_addr.  The declaration causes
   problems in s390 though, so we define __tls_get_addr here to avoid declaring
   __tls_get_addr again.  */
#  define __tls_get_addr __tls_get_addr

extern void *__tls_get_addr (tls_index *ti) attribute_hidden;
/* Make a temporary alias of __tls_get_addr to remove the hidden
   attribute.  Then export __tls_get_addr as __tls_get_addr_internal
   for use from libc.  We do not want to export __tls_get_addr, but we
   do need to use it from libc when looking up the address of a TLS
   variable. We don't use __tls_get_offset because it requires r12 to
   be setup and that might not always be true. Either way it's more
   optimal to use __tls_get_addr directly (that's what
   __tls_get_offset does anyways).  */
strong_alias (__tls_get_addr, __tls_get_addr_internal_tmp);
versioned_symbol (ld, __tls_get_addr_internal_tmp,
		  __tls_get_addr_internal, GLIBC_PRIVATE);

/* The special thing about the s390 TLS ABI is that we do not have the
   standard __tls_get_addr function but the __tls_get_offset function
   which differs in two important aspects:
   1) __tls_get_offset gets a got offset instead of a pointer to the
      tls_index structure
   2) __tls_get_offset returns the offset of the requested variable to
      the thread descriptor instead of a pointer to the variable.
 */
#  ifdef __s390x__
__asm__("\n\
	.text\n\
	.globl __tls_get_offset\n\
	.type __tls_get_offset, @function\n\
	.align 4\n\
__tls_get_offset:\n\
	la	%r2,0(%r2,%r12)\n\
	jg	__tls_get_addr\n\
");
#  elif defined __s390__
__asm__("\n\
	.text\n\
	.globl __tls_get_offset\n\
	.type __tls_get_offset, @function\n\
	.align 4\n\
__tls_get_offset:\n\
	basr	%r3,0\n\
0:	la	%r2,0(%r2,%r12)\n\
	l	%r4,1f-0b(%r3)\n\
	b	0(%r4,%r3)\n\
1:	.long	__tls_get_addr - 0b\n\
");
#  endif
# else /* IS_IN (rtld) */
extern void *__tls_get_addr_internal (tls_index *ti);
# endif /* !IS_IN (rtld) */

# define GET_ADDR_OFFSET \
  (ti->ti_offset - (unsigned long) __builtin_thread_pointer ())

/* Use the privately exported __tls_get_addr_internal instead of
   __tls_get_offset in order to avoid the __tls_get_offset special
   linkage requiring the GOT pointer to be set up in r12.  The
   compiler will take care of setting up r12 only if itself issued the
   __tls_get_offset call.  */
# define __TLS_GET_ADDR(__ti)					\
  ({ __tls_get_addr_internal (__ti)				\
      + (unsigned long) __builtin_thread_pointer (); })

#endif
