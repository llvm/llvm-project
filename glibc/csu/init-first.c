/* Initialization code run first thing by the ELF startup code.  Common version
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sysdep.h>
#include <fpu_control.h>
#include <sys/param.h>
#include <sys/types.h>
#include <libc-internal.h>

#include <ldsodefs.h>

/* Remember the command line argument and enviroment contents for
   later calls of initializers for dynamic libraries.  */
int __libc_argc attribute_hidden;
char **__libc_argv attribute_hidden;


void
__libc_init_first (int argc, char **argv, char **envp)
{
#ifdef SHARED
  /* For DSOs we do not need __libc_init_first but an ELF constructor.  */
}

static void __attribute__ ((constructor))
_init_first (int argc, char **argv, char **envp)
{
#endif

  /* Make sure we don't initialize twice.  */
#ifdef SHARED
  if (__libc_initial)
    {
      /* Set the FPU control word to the proper default value if the
	 kernel would use a different value.  */
      if (__fpu_control != GLRO(dl_fpu_control))
	__setfpucw (__fpu_control);
    }
#endif

  /* Save the command-line arguments.  */
  __libc_argc = argc;
  __libc_argv = argv;
  __environ = envp;

#ifndef SHARED
  /* First the initialization which normally would be done by the
     dynamic linker.  */
  _dl_non_dynamic_init ();
#endif

  __init_misc (argc, argv, envp);
}

/* This function is defined here so that if this file ever gets into
   ld.so we will get a link error.  Having this file silently included
   in ld.so causes disaster, because the _init_first definition above
   will cause ld.so to gain an ELF constructor, which is not a cool
   thing. */

extern void _dl_start (void) __attribute__ ((noreturn));

void
_dl_start (void)
{
  abort ();
}
