/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#include <pthreadP.h>
#include <stdlib.h>
#include <tls.h>
#include <unistd.h>

/* Default thread attributes for the case when the user does not
   provide any.  */
union pthread_attr_transparent __default_pthread_attr;
libc_hidden_data_def (__default_pthread_attr)

/* Mutex protecting __default_pthread_attr.  */
int __default_pthread_attr_lock = LLL_LOCK_INITIALIZER;
libc_hidden_data_def (__default_pthread_attr_lock)
