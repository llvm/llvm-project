/* Helper code for POSIX timer implementation on NPTL.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Kaz Kylheku <kaz@ashi.footprints.net>.

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

#include <internaltypes.h>
#include <string.h>

/* Compare two pthread_attr_t thread attributes for exact equality.
   Returns 1 if they are equal, otherwise zero if they are not equal
   or contain illegal values.  This version is NPTL-specific for
   performance reason.  One could use the access functions to get the
   values of all the fields of the attribute structure.  */
static inline int
thread_attr_compare (const pthread_attr_t *left, const pthread_attr_t *right)
{
  struct pthread_attr *ileft = (struct pthread_attr *) left;
  struct pthread_attr *iright = (struct pthread_attr *) right;

  return (ileft->flags == iright->flags
	  && ileft->schedpolicy == iright->schedpolicy
	  && (ileft->schedparam.sched_priority
	      == iright->schedparam.sched_priority)
	  && ileft->guardsize == iright->guardsize
	  && ileft->stackaddr == iright->stackaddr
	  && ileft->stacksize == iright->stacksize
	  && ((ileft->cpuset == NULL && iright->cpuset == NULL)
	      || (ileft->cpuset != NULL && iright->cpuset != NULL
		  && ileft->cpusetsize == iright->cpusetsize
		  && memcmp (ileft->cpuset, iright->cpuset,
			     ileft->cpusetsize) == 0)));
}

#endif	/* timer_routines.h */
