/* mmap - map files or devices into memory.  Linux/s390 version.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#ifndef MMAP_S390_INTERNAL_H
# define MMAP_S390_INTERNAL_H

#define MMAP_CALL(__nr, __addr, __len, __prot, __flags, __fd, __offset)	\
  ({									\
    long int __args[6] = { (long int) (__addr), (long int) (__len),	\
			   (long int) (__prot), (long int) (__flags),	\
			   (long int) (__fd), (long int) (__offset) };	\
    INLINE_SYSCALL_CALL (__nr, __args);					\
  })

#include_next <mmap_internal.h>

#endif
