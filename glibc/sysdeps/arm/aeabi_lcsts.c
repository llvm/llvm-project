/* Link-time constants for ARM EABI.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   In addition to the permissions in the GNU Lesser General Public
   License, the Free Software Foundation gives you unlimited
   permission to link the compiled version of this file with other
   programs, and to distribute those programs without any restriction
   coming from the use of this file. (The GNU Lesser General Public
   License restrictions do apply in other respects; for example, they
   cover modification of the file, and distribution when not linked
   into another program.)

   Note that people who make modified versions of this file are not
   obligated to grant this special exception for their modified
   versions; it is their choice whether to do so. The GNU Lesser
   General Public License gives permission to release a modified
   version without this exception; this exception also makes it
   possible to release a modified version which carries forward this
   exception.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* The ARM EABI requires that we provide ISO compile-time constants as
   link-time constants.  Some portable applications may reference these.  */

#include <errno.h>
#include <limits.h>
#include <locale.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <time.h>

#define eabi_constant2(X,Y) const int __aeabi_##X attribute_hidden = Y
#define eabi_constant(X) const int __aeabi_##X attribute_hidden = X

eabi_constant (EDOM);
eabi_constant (ERANGE);
eabi_constant (EILSEQ);

eabi_constant (MB_LEN_MAX);

eabi_constant (LC_COLLATE);
eabi_constant (LC_CTYPE);
eabi_constant (LC_MONETARY);
eabi_constant (LC_NUMERIC);
eabi_constant (LC_TIME);
eabi_constant (LC_ALL);

/* The value of __aeabi_JMP_BUF_SIZE is the number of doublewords in a
   jmp_buf.  */
eabi_constant2 (JMP_BUF_SIZE, sizeof (jmp_buf) / 8);

eabi_constant (SIGABRT);
eabi_constant (SIGFPE);
eabi_constant (SIGILL);
eabi_constant (SIGINT);
eabi_constant (SIGSEGV);
eabi_constant (SIGTERM);

eabi_constant2 (IOFBF, _IOFBF);
eabi_constant2 (IOLBF, _IOLBF);
eabi_constant2 (IONBF, _IONBF);
eabi_constant (BUFSIZ);
eabi_constant (FOPEN_MAX);
eabi_constant (TMP_MAX);
eabi_constant (FILENAME_MAX);
eabi_constant (L_tmpnam);

FILE *__aeabi_stdin attribute_hidden;
FILE *__aeabi_stdout attribute_hidden;
FILE *__aeabi_stderr attribute_hidden;

static void __attribute__ ((used))
setup_aeabi_stdio (void)
{
  __aeabi_stdin = stdin;
  __aeabi_stdout = stdout;
  __aeabi_stderr = stderr;
}

static void (*fp) (void) __attribute__ ((used, section (".preinit_array")))
  = setup_aeabi_stdio;

eabi_constant (CLOCKS_PER_SEC);
