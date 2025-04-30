/* System dependent pieces of sysconf; Mach version
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <mach.h>
#include <hurd.h>
#include <sys/sysinfo.h>


/* Return the number of processors configured on the system. */
int
__get_nprocs_conf (void)
{
  struct host_basic_info hbi;
  kern_return_t err;
  mach_msg_type_number_t cnt = HOST_BASIC_INFO_COUNT;

  err = __host_info (__mach_host_self (), HOST_BASIC_INFO,
		     (host_info_t) &hbi, &cnt);
  if (err)
    return __hurd_fail (err);
  else if (cnt != HOST_BASIC_INFO_COUNT)
    return __hurd_fail (EIEIO);

  return hbi.max_cpus;
}
libc_hidden_def (__get_nprocs_conf)
weak_alias (__get_nprocs_conf, get_nprocs_conf)

/* Return the number of processors currently available on the system. */
int
__get_nprocs (void)
{
  struct host_basic_info hbi;
  kern_return_t err;
  mach_msg_type_number_t cnt = HOST_BASIC_INFO_COUNT;

  err = __host_info (__mach_host_self (), HOST_BASIC_INFO,
		     (host_info_t) &hbi, &cnt);
  if (err)
    return __hurd_fail (err);
  else if (cnt != HOST_BASIC_INFO_COUNT)
    return __hurd_fail (EIEIO);

  return hbi.avail_cpus;
}
libc_hidden_def (__get_nprocs)
weak_alias (__get_nprocs, get_nprocs)

/* Return the number of physical pages on the system. */
long int
__get_phys_pages (void)
{
  struct host_basic_info hbi;
  kern_return_t err;
  mach_msg_type_number_t cnt = HOST_BASIC_INFO_COUNT;

  err = __host_info (__mach_host_self (), HOST_BASIC_INFO,
		     (host_info_t) &hbi, &cnt);
  if (err)
    return __hurd_fail (err);
  else if (cnt != HOST_BASIC_INFO_COUNT)
    return __hurd_fail (EIEIO);

  return hbi.memory_size / __vm_page_size;
}
libc_hidden_def (__get_phys_pages)
weak_alias (__get_phys_pages, get_phys_pages)

/* Return the number of available physical pages */
long int
__get_avphys_pages (void)
{
  vm_statistics_data_t vs;
  kern_return_t err;

#ifdef HOST_VM_INFO
  {
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    err = __host_info (__mach_host_self (), HOST_VM_INFO,
		       (host_info_t) &vs, &count);
    if (!err && count < HOST_VM_INFO_COUNT)
      err = EGRATUITOUS;
  }
#else
  err = __vm_statistics (__mach_task_self (), &vs);
#endif
  if (err)
    return __hurd_fail (err);

  return vs.free_count;
}
libc_hidden_def (__get_avphys_pages)
weak_alias (__get_avphys_pages, get_avphys_pages)
