/* Macros for copying by pages; used in memcpy, memmove.
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

/* The macro PAGE_COPY_FWD_MAYBE defined in memcopy.h is used in memmove if the
   PAGE_COPY_THRESHOLD macro is set to a non-zero value.  The default is 0,
   that is copying by pages is not implemented.

   System-specific pagecopy.h files that want to support page copying should
   define these macros:

   PAGE_COPY_THRESHOLD
   -- A non-zero minimum size for which virtual copying by pages is worthwhile.

   PAGE_SIZE
   -- Size of a page.

   PAGE_COPY_FWD (dstp, srcp, nbytes_left, nbytes)
   -- Macro to perform the virtual copy operation.
   The pointers will be aligned to PAGE_SIZE bytes.
*/

#define PAGE_COPY_THRESHOLD 0
