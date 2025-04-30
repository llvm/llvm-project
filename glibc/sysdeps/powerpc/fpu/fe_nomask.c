/* Procedure definition for FE_NOMASK_ENV.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <fenv_libc.h>
#include <errno.h>

/* This is a generic stub. An OS specific override is required to set
   the FE0/FE1 bits in the MSR.  MSR update is privileged, so this will
   normally involve a syscall.  */

const fenv_t *
__fe_nomask_env_priv (void)
{
  __set_errno (ENOSYS);
  return FE_ENABLED_ENV;
}
stub_warning (__fe_nomask_env_priv)
