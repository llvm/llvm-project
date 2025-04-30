/* Get a host configuration item kept as the whole contents of a file.
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

#include <fcntl.h>
#include <hurd.h>
#include <hurd/lookup.h>
#include "hurdhost.h"
#include <string.h>

ssize_t
_hurd_get_host_config (const char *item, char *buf, size_t buflen)
{
  error_t err;
  char *data;
  mach_msg_type_number_t nread, more;
  file_t config;

  err = __hurd_file_name_lookup (&_hurd_ports_use, &__getdport, 0,
				 item, O_RDONLY, 0, &config);
  switch (err)
    {
    case 0:			/* Success; read file contents below.  */
      break;

    case ENOENT:		/* ? Others?  All errors? */
      /* The file does not exist, so no value has been set.  Rather than
	 causing gethostname et al to fail with ENOENT, give an empty value
	 as other systems do before sethostname has been called.  */
      if (buflen != 0)
	*buf = '\0';
      return 0;

    default:
      return __hurd_fail (err);
    }

  data = buf;
  nread = buflen;
  err = __io_read (config, &data, &nread, -1, buflen);
  if (! err)
    /* Check if there is more in the file we didn't read.  */
    err = __io_readable (config, &more);
  __mach_port_deallocate (__mach_task_self (), config);
  if (err)
    return __hurd_fail (err);
  if (data != buf)
    {
      memcpy (buf, data, nread);
      __vm_deallocate (__mach_task_self (), (vm_address_t) data, nread);
    }

  /* If the file is empty, give an empty value.  */
  if (nread == 0 && more == 0)
    {
      if (buflen != 0)
	*buf = '\0';
      return 0;
    }

  /* Remove newlines in case someone wrote the file by hand.  */
  while (nread > 0 && buf[nread - 1] == '\n')
    buf[--nread] = '\0';

  /* Null-terminate the result if there is enough space.  */
  if (nread < buflen)
    buf[nread] = '\0';
  else
    if (nread != 0 && buf[nread - 1] != '\0')
      more = 1;

  if (more)
    /* If we didn't read the whole file, tell the caller to use a bigger
       buffer next time.  */
    return __hurd_fail (ENAMETOOLONG);

  return nread;
}
