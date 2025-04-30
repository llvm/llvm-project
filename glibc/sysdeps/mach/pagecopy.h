/* Macros for copying by pages; used in memcpy, memmove.  Mach version.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

/* Threshold at which vm_copy is more efficient than well-optimized copying
   by words.  */
#define PAGE_COPY_THRESHOLD		(16384)

#define PAGE_SIZE		__vm_page_size
#define PAGE_COPY_FWD(dstp, srcp, nbytes_left, nbytes)			      \
  ((nbytes_left) = ((nbytes)						      \
		    - (__vm_copy (__mach_task_self (),			      \
				  (vm_address_t) srcp, trunc_page (nbytes),   \
				  (vm_address_t) dstp) == KERN_SUCCESS	      \
		       ? trunc_page (nbytes)				      \
		       : 0)))
