/* Code to enable profiling at program startup.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   In addition to the permissions in the GNU Lesser General Public
   License, the Free Software Foundation gives you unlimited
   permission to link the compiled version of this file with other
   programs, and to distribute those programs without any restriction
   coming from the use of this file.  (The GNU Lesser General Public
   License restrictions do apply in other respects; for example, they
   cover modification of the file, and distribution when not linked
   into another program.)

   Note that people who make modified versions of this file are not
   obligated to grant this special exception for their modified
   versions; it is their choice whether to do so.  The GNU Lesser
   General Public License gives permission to release a modified
   version without this exception; this exception also makes it
   possible to release a modified version which carries forward this
   exception.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sys/types.h>
#include <sys/gmon.h>
#include <stdlib.h>
#include <unistd.h>
#include <elf-initfini.h>
#define __ASSEMBLY__
#include <entry.h>

/* Beginning and end of our code segment. We cannot declare them
   as the external functions since we want the addresses of those
   labels. Taking the address of a function may have different
   meanings on different platforms. */
#ifdef ENTRY_POINT_DECL
ENTRY_POINT_DECL(extern)
#else
extern char ENTRY_POINT[];
#endif
extern char etext[];

#ifndef TEXT_START
# ifdef ENTRY_POINT_DECL
#  define TEXT_START ENTRY_POINT
# else
#  define TEXT_START &ENTRY_POINT
# endif
#endif

#if !ELF_INITFINI
/* Instead of defining __gmon_start__ globally in gcrt1.o, we make it
   static and just put a pointer to it into the .preinit_array
   section.  */
# define GMON_START_ARRAY_SECTION ".preinit_array"
#endif

#ifdef GMON_START_ARRAY_SECTION
static void __gmon_start__ (void);
static void (*const gmon_start_initializer) (void)
  __attribute__ ((used, section (GMON_START_ARRAY_SECTION))) = &__gmon_start__;
static
#else
/* We cannot use the normal constructor mechanism to call
   __gmon_start__ because gcrt1.o appears before crtbegin.o in the link.
   Instead crti.o calls it specially.  */
extern void __gmon_start__ (void);
#endif

void
__gmon_start__ (void)
{
  /* Protect from being called more than once.  Since crti.o is linked
     into every shared library, each of their init functions will call us.  */
  static int called;

  if (called)
    return;

  called = 1;

  /* Start keeping profiling records.  */
  __monstartup ((u_long) TEXT_START, (u_long) &etext);

  /* Call _mcleanup before exiting; it will write out gmon.out from the
     collected data.  */
  atexit (&_mcleanup);
}
