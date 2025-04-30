/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#ifndef _ATOMIC_MACHINE_H
#define _ATOMIC_MACHINE_H	1

/* We have by default no support for atomic operations.  So define
   them non-atomic.  If this is a problem somebody will have to come
   up with real definitions.  */

/* The only basic operation needed is compare and exchange.  */
#define atomic_compare_and_exchange_val_acq(mem, newval, oldval) \
  ({ __typeof (mem) __gmemp = (mem);				      \
     __typeof (*mem) __gret = *__gmemp;				      \
     __typeof (*mem) __gnewval = (newval);			      \
								      \
     if (__gret == (oldval))					      \
       *__gmemp = __gnewval;					      \
     __gret; })

#define atomic_compare_and_exchange_bool_acq(mem, newval, oldval) \
  ({ __typeof (mem) __gmemp = (mem);				      \
     __typeof (*mem) __gnewval = (newval);			      \
								      \
     *__gmemp == (oldval) ? (*__gmemp = __gnewval, 0) : 1; })

#endif	/* atomic-machine.h */
