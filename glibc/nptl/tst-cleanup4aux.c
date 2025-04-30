/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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

#include <pthread.h>
#include <shlib-compat.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern void _pthread_cleanup_push (struct _pthread_cleanup_buffer *__buffer,
                                   void (*__routine) (void *),
                                   void *__arg);
compat_symbol_reference (libpthread, _pthread_cleanup_push,
                         _pthread_cleanup_push, GLIBC_2_0);
extern void _pthread_cleanup_pop (struct _pthread_cleanup_buffer *__buffer,
                                  int __execute);
compat_symbol_reference (libpthread, _pthread_cleanup_pop,
                         _pthread_cleanup_pop, GLIBC_2_0);

extern void clh (void *arg);
extern void fn0 (void);
extern void fn1 (void);
extern void fn5 (void);
extern void fn7 (void);
extern void fn9 (void);


static __attribute__((noinline)) void
fn3 (void)
{
  /* This is the old LinuxThreads pthread_cleanup_{push,pop}.  */
     struct _pthread_cleanup_buffer b;
  _pthread_cleanup_push (&b, clh, (void *) 4l);

  fn0 ();

  _pthread_cleanup_pop (&b, 1);
}


static __attribute__((noinline)) void
fn4 (void)
{
  pthread_cleanup_push (clh, (void *) 5l);

  fn3 ();

  pthread_cleanup_pop (1);
}


void
fn5 (void)
{
  /* This is the old LinuxThreads pthread_cleanup_{push,pop}.  */
     struct _pthread_cleanup_buffer b;
  _pthread_cleanup_push (&b, clh, (void *) 6l);

  fn4 ();

  _pthread_cleanup_pop (&b, 1);
}


static __attribute__((noinline)) void
fn6 (void)
{
  pthread_cleanup_push (clh, (void *) 7l);

  fn0 ();

  pthread_cleanup_pop (1);
}


void
fn7 (void)
{
  /* This is the old LinuxThreads pthread_cleanup_{push,pop}.  */
     struct _pthread_cleanup_buffer b;
  _pthread_cleanup_push (&b, clh, (void *) 8l);

  fn6 ();

  _pthread_cleanup_pop (&b, 1);
}


static __attribute__((noinline)) void
fn8 (void)
{
  pthread_cleanup_push (clh, (void *) 9l);

  fn1 ();

  pthread_cleanup_pop (1);
}


void
fn9 (void)
{
  /* This is the old LinuxThreads pthread_cleanup_{push,pop}.  */
     struct _pthread_cleanup_buffer b;
  _pthread_cleanup_push (&b, clh, (void *) 10l);

  fn8 ();

  _pthread_cleanup_pop (&b, 1);
}
