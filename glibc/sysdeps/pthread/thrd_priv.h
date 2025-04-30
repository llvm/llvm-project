/* Internal C11 threads definitions.
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

#ifndef THRD_PRIV_H
# define THRD_PRIV_H

#include <features.h>
#include <threads.h>
#include <errno.h>
#include "pthreadP.h"	/* For pthread_{mutex,cond}_t definitions.  */

static __always_inline int
thrd_err_map (int err_code)
{
  switch (err_code)
  {
    case 0:
      return thrd_success;
    case ENOMEM:
      return thrd_nomem;
    case ETIMEDOUT:
      return thrd_timedout;
    case EBUSY:
      return thrd_busy;
    default:
      return thrd_error;
  }
}

#endif
