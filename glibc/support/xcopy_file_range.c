/* copy_file_range with error checking.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <support/support.h>
#include <support/xunistd.h>
#include <support/check.h>

ssize_t
xcopy_file_range (int infd, off64_t *pinoff, int outfd, off64_t *poutoff,
		  size_t length, unsigned int flags)
{
  ssize_t status = support_copy_file_range (infd, pinoff, outfd,
					    poutoff, length, flags);
  if (status == -1)
    FAIL_EXIT1 ("cannot copy file: %m\n");
  return status;
}
