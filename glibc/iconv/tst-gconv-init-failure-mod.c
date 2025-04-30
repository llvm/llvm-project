/* Test gconv module for tst-gconv-init-failure.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <gconv.h>
#include <support/check.h>
#include <support/support.h>

int
gconv (struct __gconv_step *step,
       struct __gconv_step_data *data,
       const unsigned char **inptrp,
       const unsigned char *inend,
       unsigned char **outbufstart, size_t *irreversible,
       int do_flush, int consume_incomplete)
{
  FAIL_EXIT1 ("gconv called");
  return __GCONV_INTERNAL_ERROR;
}

int
gconv_init (struct __gconv_step *ignored)
{
  write_message ("info: gconv_init called, returning error\n");
  errno = ENOMEM;
  return __GCONV_NOMEM;
}

int
gconv_end (struct __gconv_step *ignored)
{
  FAIL_EXIT1 ("gconv_end called");
  return __GCONV_INTERNAL_ERROR;
}
