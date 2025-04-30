/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2005.

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

/* We test the code undef conditions outside of glibc.  */
#undef _LIBC

#include <pthread.h>

pthread_mutex_t mtx_normal = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mtx_recursive = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
pthread_mutex_t mtx_errorchk = PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP;
pthread_mutex_t mtx_adaptive = PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP;
pthread_rwlock_t rwl_normal = PTHREAD_RWLOCK_INITIALIZER;
pthread_rwlock_t rwl_writer
  = PTHREAD_RWLOCK_WRITER_NONRECURSIVE_INITIALIZER_NP;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static int
do_test (void)
{
  if (mtx_normal.__data.__kind != PTHREAD_MUTEX_TIMED_NP)
    return 1;
  if (mtx_recursive.__data.__kind != PTHREAD_MUTEX_RECURSIVE_NP)
    return 2;
  if (mtx_errorchk.__data.__kind != PTHREAD_MUTEX_ERRORCHECK_NP)
    return 3;
  if (mtx_adaptive.__data.__kind != PTHREAD_MUTEX_ADAPTIVE_NP)
    return 4;
  if (rwl_normal.__data.__flags != PTHREAD_RWLOCK_PREFER_READER_NP)
    return 5;
  if (rwl_writer.__data.__flags
      != PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP)
    return 6;
  /* <libc-lock.h> __libc_rwlock_init definition for libc.so
     relies on PTHREAD_RWLOCK_INITIALIZER being all zeros.  If
     that ever changes, <libc-lock.h> needs updating.  */
  size_t i;
  for (i = 0; i < sizeof (rwl_normal); i++)
    if (((char *) &rwl_normal)[i] != '\0')
      return 7;
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
