/* HPPA internal mutex struct definitions.
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

#ifndef _THREAD_MUTEX_INTERNAL_H
#define _THREAD_MUTEX_INTERNAL_H 1

struct __pthread_mutex_s
{
  int __lock __LOCK_ALIGNMENT;
  unsigned int __count;
  int __owner;
  /* KIND must stay at this position in the structure to maintain
     binary compatibility with static initializers.  */
  int __kind;
  /* The old 4-word 16-byte aligned lock. This is initalized
     to all ones by the Linuxthreads PTHREAD_MUTEX_INITIALIZER.
     Unused in NPTL.  */
  int __glibc_compat_padding[4];
  /* In the old structure there are 4 words left due to alignment.
     In NPTL two words are used.  */
  unsigned int __nusers;
  __extension__ union
  {
    int __spins;
    __pthread_slist_t __list;
  };
  /* Two more words are left before the NPTL
     pthread_mutex_t is larger than Linuxthreads.  */
  int __glibc_reserved1;
  int __glibc_reserved2;
};

#define __PTHREAD_MUTEX_HAVE_PREV       0

#define __PTHREAD_MUTEX_INITIALIZER(__kind) \
  0, 0, 0, __kind, { 0, 0, 0, 0 }, 0, { 0 }, 0, 0

#endif
