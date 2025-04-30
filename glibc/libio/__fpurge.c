/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

#include <stdio_ext.h>
#include "libioP.h"

void
__fpurge (FILE *fp)
{
  if (fp->_mode > 0)
    {
      /* Wide-char stream.  */
      if (_IO_in_backup (fp))
	_IO_free_wbackup_area (fp);

      fp->_wide_data->_IO_read_end = fp->_wide_data->_IO_read_ptr;
      fp->_wide_data->_IO_write_ptr = fp->_wide_data->_IO_write_base;
    }
  else
    {
      /* Byte stream.  */
      if (_IO_in_backup (fp))
	_IO_free_backup_area (fp);

      fp->_IO_read_end = fp->_IO_read_ptr;
      fp->_IO_write_ptr = fp->_IO_write_base;
    }
}
