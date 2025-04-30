/* Helper code for POSIX timer implementation on Hurd.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#ifndef _TIMER_ROUTINES_H
#define _TIMER_ROUTINES_H	1

#include <bits/pthreadtypes.h>

/* Compare two pthread_attr_t thread attributes for exact equality.
   Returns 1 if they are equal, otherwise zero if they are not equal
   or contain illegal values.  This version is Hurd-specific for
   performance reason.  One could use the access functions to get the
   values of all the fields of the attribute structure.  */
static inline int
thread_attr_compare (const pthread_attr_t * left, const pthread_attr_t * right)
{
  struct __pthread_attr *ileft = (struct __pthread_attr *) left;
  struct __pthread_attr *iright = (struct __pthread_attr *) right;

  return ileft->__schedparam.sched_priority
	   == iright->__schedparam.sched_priority
	 && ileft->__stackaddr == iright->__stackaddr
	 && ileft->__stacksize == iright->__stacksize
	 && ileft->__guardsize == iright->__guardsize
	 && ileft->__detachstate == iright->__detachstate
	 && ileft->__inheritsched == iright->__inheritsched
	 && ileft->__contentionscope == iright->__contentionscope
	 && ileft->__schedpolicy == iright->__schedpolicy;
}

#endif /* timer_routines.h */
