/* Machine-dependent SIGPROF signal handler.  "Generic" version w/ sigcontext
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

/* In many Unix systems signal handlers are called like this
   and the interrupted PC is easily findable in the `struct sigcontext'.  */

#ifdef SA_SIGINFO
#include <sigcontextinfo.h>

static void
__profil_counter (int signr, siginfo_t *info, void *ctx)
{
  profil_count (sigcontext_get_pc (ctx));
}
#else
static void
__profil_counter (int signr, int code, struct sigcontext *scp)
{
  profil_count ((uintptr_t) scp->sc_pc);
}
#endif
