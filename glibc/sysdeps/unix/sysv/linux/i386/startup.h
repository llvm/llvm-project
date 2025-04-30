/* Linux/i386 definitions of functions used by static libc main startup.
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

#if BUILD_PIE_DEFAULT
/* Can't use "call *%gs:SYSINFO_OFFSET" during statup in static PIE.  */
# define I386_USE_SYSENTER 0

# include <sysdep.h>
# include <abort-instr.h>

__attribute__ ((__noreturn__))
static inline void
_startup_fatal (const char *message __attribute__ ((unused)))
{
  /* This is only called very early during startup in static PIE.
     FIXME: How can it be improved?  */
  ABORT_INSTRUCTION;
  __builtin_unreachable ();
}

static inline uid_t
startup_getuid (void)
{
  return (uid_t) INTERNAL_SYSCALL_CALL (getuid32);
}

static inline uid_t
startup_geteuid (void)
{
  return (uid_t) INTERNAL_SYSCALL_CALL (geteuid32);
}

static inline gid_t
startup_getgid (void)
{
  return (gid_t) INTERNAL_SYSCALL_CALL (getgid32);
}

static inline gid_t
startup_getegid (void)
{
  return (gid_t) INTERNAL_SYSCALL_CALL (getegid32);
}
#else
# include_next <startup.h>
#endif
