/* Declarations and macros for the basic Mach things set at startup.
   Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

#ifndef	_MACH_INIT_H

#define	_MACH_INIT_H	1

#include <mach/mach_types.h>

/* Return the current task's task port.  */
extern mach_port_t __mach_task_self (void);
extern mach_port_t mach_task_self (void);

/* This cache is initialized at startup.  */
extern mach_port_t __mach_task_self_;
#define __mach_task_self()	(__mach_task_self_ + 0)	/* Not an lvalue.  */
#define mach_task_self()	(__mach_task_self ())

/* This cache is initialized at startup.  */
extern mach_port_t __mach_host_self_;
#define __mach_host_self()	(__mach_host_self_ + 0)	/* Not an lvalue.  */
#define mach_host_self()	(__mach_host_self ())

/* Kernel page size.  */
extern vm_size_t __vm_page_size;
extern vm_size_t vm_page_size;

/* Round the address X up to a page boundary.  */
#define round_page(x)	\
  ((((vm_offset_t) (x) + __vm_page_size - 1) / __vm_page_size) * \
   __vm_page_size)

/* Truncate the address X down to a page boundary.  */
#define trunc_page(x)	\
  ((((vm_offset_t) (x)) / __vm_page_size) * __vm_page_size)

#endif	/* mach_init.h */
