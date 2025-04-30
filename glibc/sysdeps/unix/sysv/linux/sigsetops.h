/* __sigset_t manipulators.  Linux version.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#ifndef _SIGSETOPS_H
#define _SIGSETOPS_H 1

#include <signal.h>
#include <limits.h>
#include <libc-pointer-arith.h>

/* Return a mask that includes the bit for SIG only.  */
#define __sigmask(sig) \
  (1UL << (((sig) - 1) % ULONG_WIDTH))

/* Return the word index for SIG.  */
static inline unsigned long int
__sigword (int sig)
{
  return (sig - 1) / ULONG_WIDTH;
}

/* Linux sig* functions only handle up to __NSIG_WORDS words instead of
   full _SIGSET_NWORDS sigset size.  The signal numbers are 1-based, and
   bit 0 of a signal mask is for signal 1.  */
#define __NSIG_WORDS (ALIGN_UP ((_NSIG - 1), ULONG_WIDTH) / ULONG_WIDTH)
_Static_assert (__NSIG_WORDS <= _SIGSET_NWORDS,
		"__NSIG_WORDS > _SIGSET_WORDS");

/* This macro is used on syscall that takes a sigset_t to specify the expected
   size in bytes.  As for glibc, kernel sigset is implemented as an array of
   unsigned long.  */
#define __NSIG_BYTES (__NSIG_WORDS * (ULONG_WIDTH / UCHAR_WIDTH))

static inline void
__sigemptyset (sigset_t *set)
{
  int cnt = __NSIG_WORDS;
  while (--cnt >= 0)
   set->__val[cnt] = 0;
}

static inline void
__sigfillset (sigset_t *set)
{
  int cnt = __NSIG_WORDS;
  while (--cnt >= 0)
   set->__val[cnt] = ~0UL;
}

static inline int
__sigisemptyset (const sigset_t *set)
{
  int cnt = __NSIG_WORDS;
  int ret = set->__val[--cnt];
  while (ret == 0 && --cnt >= 0)
    ret = set->__val[cnt];
  return ret == 0;
}

static inline void
__sigandset (sigset_t *dest, const sigset_t *left, const sigset_t *right)
{
  int cnt = __NSIG_WORDS;
  while (--cnt >= 0)
    dest->__val[cnt] = left->__val[cnt] & right->__val[cnt];
}

static inline void
__sigorset (sigset_t *dest, const sigset_t *left, const sigset_t *right)
{
  int cnt = __NSIG_WORDS;
  while (--cnt >= 0)
    dest->__val[cnt] = left->__val[cnt] | right->__val[cnt];
}

static inline int
__sigismember (const sigset_t *set, int sig)
{
  unsigned long int mask = __sigmask (sig);
  unsigned long int word = __sigword (sig);
  return set->__val[word] & mask ? 1 : 0;
}

static inline void
__sigaddset (sigset_t *set, int sig)
{
  unsigned long int mask = __sigmask (sig);
  unsigned long int word = __sigword (sig);
  set->__val[word] |= mask;
}

static inline void
__sigdelset (sigset_t *set, int sig)
{
  unsigned long int mask = __sigmask (sig);
  unsigned long int word = __sigword (sig);
  set->__val[word] &= ~mask;
}

#endif /* bits/sigsetops.h */
