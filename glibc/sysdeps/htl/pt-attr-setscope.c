/* pthread_attr_setscope.  Generic version.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <pthread.h>
#include <pt-internal.h>

int
__pthread_attr_setscope (pthread_attr_t *attr, int contentionscope)
{
  if (contentionscope == __pthread_default_attr.__contentionscope)
    {
      attr->__contentionscope = contentionscope;
      return 0;
    }

  switch (contentionscope)
    {
    case PTHREAD_SCOPE_PROCESS:
    case PTHREAD_SCOPE_SYSTEM:
      return ENOTSUP;
    default:
      return EINVAL;
    }
}

weak_alias (__pthread_attr_setscope, pthread_attr_setscope);
