/* Low-level statistical profiling support function.  Linux version.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <signal.h>
#include <sigcontextinfo.h>

/* sa_sigaction signature to use along SA_SIGINFO.  */
static void
__profil_counter (int signo, siginfo_t *info, void *ctx)
{
  profil_count (sigcontext_get_pc (ctx));

  /* This is a hack to prevent the compiler from implementing the
     above function call as a sibcall.  The sibcall would overwrite
     the signal context.  */
  asm volatile ("");
}
