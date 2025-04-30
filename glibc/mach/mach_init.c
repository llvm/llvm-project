/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include <mach.h>
#include <mach/mig_support.h>
#include <unistd.h>

mach_port_t __mach_task_self_;
mach_port_t __mach_host_self_;
vm_size_t __vm_page_size = 0;	/* Must be data not bss for weak alias.  */
weak_alias (__vm_page_size, vm_page_size)

#ifdef NDR_DEF_HEADER
/* This defines NDR_record, which the MiG-generated stubs use. XXX namespace */
# include NDR_DEF_HEADER
#endif

void
__mach_init (void)
{
  kern_return_t err;

  __mach_task_self_ = (__mach_task_self) ();
  __mach_host_self_ = (__mach_host_self) ();
  __mig_init (0);

#ifdef HAVE_HOST_PAGE_SIZE
  if (err = __host_page_size (__mach_host_self (), &__vm_page_size))
    _exit (err);
#else
  {
    vm_statistics_data_t stats;
    if (err = __vm_statistics (__mach_task_self (), &stats))
      _exit (err);
    __vm_page_size = stats.pagesize;
  }
#endif
}
weak_alias (__mach_init, mach_init)
