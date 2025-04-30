/* Procedure definition for FE_MASK_ENV for Linux/ppc.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <sysdep.h>
#include <sys/prctl.h>

const fenv_t *
__fe_mask_env (void)
{
  INTERNAL_SYSCALL_CALL (prctl, PR_SET_FPEXC, PR_FP_EXC_DISABLED);

  return FE_DFL_ENV;
}
