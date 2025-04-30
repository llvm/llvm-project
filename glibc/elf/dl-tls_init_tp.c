/* Completion of TCB initialization after TLS_INIT_TP.  Generic version.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <ldsodefs.h>

#if defined SHARED && defined _LIBC_REENTRANT \
    && defined __rtld_lock_default_lock_recursive
static void
rtld_lock_default_lock_recursive (void *lock)
{
  __rtld_lock_default_lock_recursive (lock);
}

static void
rtld_lock_default_unlock_recursive (void *lock)
{
  __rtld_lock_default_unlock_recursive (lock);
}
#endif

void
__tls_pre_init_tp (void)
{
#if !THREAD_GSCOPE_IN_TCB
  GL(dl_init_static_tls) = &_dl_nothread_init_static_tls;
#endif

#if defined SHARED && defined _LIBC_REENTRANT \
    && defined __rtld_lock_default_lock_recursive
  GL(dl_rtld_lock_recursive) = rtld_lock_default_lock_recursive;
  GL(dl_rtld_unlock_recursive) = rtld_lock_default_unlock_recursive;
#endif
}

void
__tls_init_tp (void)
{
}
