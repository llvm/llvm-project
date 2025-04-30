/* HPPA  internal rwlock struct definitions.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.

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
   <http://www.gnu.org/licenses/>.  */

#ifndef _RWLOCK_INTERNAL_H
#define _RWLOCK_INTERNAL_H

struct __pthread_rwlock_arch_t
{
  /* In the old Linuxthreads pthread_rwlock_t, this is the
     start of the 4-word 16-byte aligned lock structure. The
     next four words are all set to 1 by the Linuxthreads
     PTHREAD_RWLOCK_INITIALIZER. We ignore them in NPTL.  */
  int __compat_padding[4] __attribute__ ((__aligned__(16)));
  unsigned int __readers;
  unsigned int __writers;
  unsigned int __wrphase_futex;
  unsigned int __writers_futex;
  unsigned int __pad3;
  unsigned int __pad4;
  int __cur_writer;
  /* An unused word, reserved for future use. It was added
     to maintain the location of the flags from the Linuxthreads
     layout of this structure.  */
  int __reserved1;
  /* FLAGS must stay at this position in the structure to maintain
     binary compatibility.  */
  unsigned char __pad2;
  unsigned char __pad1;
  unsigned char __shared;
  unsigned char __flags;
  /* The NPTL pthread_rwlock_t is 4 words smaller than the
     Linuxthreads version. One word is in the middle of the
     structure, the other three are at the end.  */
  int __reserved2;
  int __reserved3;
  int __reserved4;
};

#define __PTHREAD_RWLOCK_INITIALIZER(__flags) \
  { 0, 0, 0, 0 }, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, __flags, 0, 0, 0

#endif
