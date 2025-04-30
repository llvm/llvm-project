/* Function descriptors.  IA64 version.
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

#ifndef dl_ia64_fptr_h
#define dl_ia64_fptr_h 1

#include <ia64intrin.h>
#include <sysdeps/generic/dl-fptr.h>

#define COMPARE_AND_SWAP(ptr, old, new)	\
  __sync_bool_compare_and_swap (ptr, old, new)

/* There are currently 123 dynamic symbols in ld.so.
   ELF_MACHINE_BOOT_FPTR_TABLE_LEN needs to be at least that big.  */
#define ELF_MACHINE_BOOT_FPTR_TABLE_LEN	200

#define ELF_MACHINE_LOAD_ADDRESS(var, symbol)	\
  asm ("movl %0 = @gprel (" #symbol ");; add %0 = %0, gp" : "=&r" (var));

/* We don't have a gcc helper to extract the plabel info.  */
#define ELF_PTR_TO_FDESC(ptr) \
  ({ union { \
       void *_ptr; \
       struct fdesc *_fdesc; \
     } _u; \
     _u._ptr = ptr; \
     _u._fdesc; \
  })

#endif /* !dl_ia64_fptr_h */
