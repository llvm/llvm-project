/* Copyright (C) 1992-2021 Free Software Foundation, Inc.
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

#include "version.h"
#include <tls.h>
#include <libc-abis.h>
#include <gnu/libc-version.h>

static const char __libc_release[] = RELEASE;
static const char __libc_version[] = VERSION;

static const char banner[] =
"GNU C Library "PKGVERSION RELEASE" release version "VERSION".\n\
Copyright (C) 2021 Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.\n\
There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A\n\
PARTICULAR PURPOSE.\n\
Compiled by GNU CC version "__VERSION__".\n"
#ifdef LIBC_ABIS_STRING
LIBC_ABIS_STRING
#endif
"For bug reporting instructions, please see:\n\
"REPORT_BUGS_TO".\n";

#include <unistd.h>

extern void __libc_print_version (void) attribute_hidden;
void
__libc_print_version (void)
{
  __write (STDOUT_FILENO, banner, sizeof banner - 1);
}

extern const char *__gnu_get_libc_release (void);
const char *
__gnu_get_libc_release (void)
{
  return __libc_release;
}
weak_alias (__gnu_get_libc_release, gnu_get_libc_release)

extern const char *__gnu_get_libc_version (void);
const char *
__gnu_get_libc_version (void)
{
  return __libc_version;
}
weak_alias (__gnu_get_libc_version, gnu_get_libc_version)

/* This function is the entry point for the shared object.
   Running the library as a program will get here.  */

extern void __libc_main (void) __attribute__ ((noreturn));
void
__libc_main (void)
{
  __libc_print_version ();
  _exit (0);
}
