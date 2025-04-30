/* Software floating-point exception handling emulation.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
   Contributed by Aldy Hernandez <aldyh@redhat.com>, 2002.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <signal.h>
#include "soft-fp.h"
#include "soft-supp.h"

/* Thread-local to store sticky exceptions.  */
__thread int __sim_exceptions_thread;
libc_hidden_tls_def (__sim_exceptions_thread);

/* By default, no exceptions should trap.  */
__thread int __sim_disabled_exceptions_thread = 0xffffffff;
libc_hidden_tls_def (__sim_disabled_exceptions_thread);

__thread int __sim_round_mode_thread;
libc_hidden_tls_def (__sim_round_mode_thread);

#if SIM_GLOBAL_COMPAT
int __sim_exceptions_global;
libc_hidden_data_def (__sim_exceptions_global);
SIM_COMPAT_SYMBOL (__sim_exceptions_global, __sim_exceptions);

int __sim_disabled_exceptions_global = 0xffffffff;
libc_hidden_data_def (__sim_disabled_exceptions_global);
SIM_COMPAT_SYMBOL (__sim_disabled_exceptions_global,
		   __sim_disabled_exceptions);

int __sim_round_mode_global;
libc_hidden_data_def (__sim_round_mode_global);
SIM_COMPAT_SYMBOL (__sim_round_mode_global, __sim_round_mode);
#endif

void
__simulate_exceptions (int x)
{
  __sim_exceptions_thread |= x;
  SIM_SET_GLOBAL (__sim_exceptions_global, __sim_exceptions_thread);
  if (x & ~__sim_disabled_exceptions_thread)
    raise (SIGFPE);
}
