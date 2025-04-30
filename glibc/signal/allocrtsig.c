/* Handle real-time signal allocation.  Generic version.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <signal.h>
#include <internal-signals.h>

/* In these variables we keep track of the used variables.  If the
   platform does not support any real-time signals we will define the
   values to some unreasonable value which will signal failing of all
   the functions below.  */
#ifdef __SIGRTMIN
static int current_rtmin = __SIGRTMIN + RESERVED_SIGRT;
static int current_rtmax = __SIGRTMAX;
#endif

/* Return number of available real-time signal with highest priority.  */
int
__libc_current_sigrtmin (void)
{
#ifdef __SIGRTMIN
  return current_rtmin;
#else
  return -1;
#endif
}
libc_hidden_def (__libc_current_sigrtmin)

/* Return number of available real-time signal with lowest priority.  */
int
__libc_current_sigrtmax (void)
{
#ifdef __SIGRTMIN
  return current_rtmax;
#else
  return -1;
#endif
}
libc_hidden_def (__libc_current_sigrtmax)

/* Allocate real-time signal with highest/lowest available
   priority.  Please note that we don't use a lock since we assume
   this function to be called at program start.  */
int
__libc_allocate_rtsig (int high)
{
#ifndef __SIGRTMIN
  return -1;
#else
  if (current_rtmin == -1 || current_rtmin > current_rtmax)
    /* We don't have any more signals available.  */
    return -1;

  return high ? current_rtmin++ : current_rtmax--;
#endif
}
