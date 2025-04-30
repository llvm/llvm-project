/* Internal support stuff for complete soft float.
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

#if defined __NO_FPRS__ && !defined _SOFT_FLOAT

# include <fenv_libc.h>

#else

# include <fenv.h>

typedef union
{
  fenv_t fenv;
  unsigned int l[2];
} fenv_union_t;

#endif

extern __thread int __sim_exceptions_thread attribute_tls_model_ie;
libc_hidden_tls_proto (__sim_exceptions_thread, tls_model ("initial-exec"));
extern __thread int __sim_disabled_exceptions_thread attribute_tls_model_ie;
libc_hidden_tls_proto (__sim_disabled_exceptions_thread,
		       tls_model ("initial-exec"));
extern __thread int __sim_round_mode_thread attribute_tls_model_ie;
libc_hidden_tls_proto (__sim_round_mode_thread, tls_model ("initial-exec"));

/* These variables were formerly global, so there are compat symbols
   for global versions as well.  */

#include <shlib-compat.h>
#define SIM_GLOBAL_COMPAT SHLIB_COMPAT (libc, GLIBC_2_3_2, GLIBC_2_19)
#if SIM_GLOBAL_COMPAT
extern int __sim_exceptions_global;
libc_hidden_proto (__sim_exceptions_global);
extern int __sim_disabled_exceptions_global ;
libc_hidden_proto (__sim_disabled_exceptions_global);
extern int __sim_round_mode_global;
libc_hidden_proto (__sim_round_mode_global);
# define SIM_COMPAT_SYMBOL(GLOBAL_NAME, NAME) \
  compat_symbol (libc, GLOBAL_NAME, NAME, GLIBC_2_3_2)
# define SIM_SET_GLOBAL(GLOBAL_VAR, THREAD_VAR) ((GLOBAL_VAR) = (THREAD_VAR))
#else
# define SIM_SET_GLOBAL(GLOBAL_VAR, THREAD_VAR) ((void) 0)
#endif

extern void __simulate_exceptions (int x) attribute_hidden;
