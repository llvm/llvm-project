/* Macros for controlling diagnostic output from the compiler.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _LIBC_DIAG_H
#define _LIBC_DIAG_H 1

/* Ignore the value of an expression when a cast to void does not
   suffice (in particular, for a call to a function declared with
   attribute warn_unused_result).  */
#define ignore_value(x) \
  ({ __typeof__ (x) __ignored_value = (x); (void) __ignored_value; })

/* The macros to control diagnostics are structured like this, rather
   than a single macro that both pushes and pops diagnostic state and
   takes the affected code as an argument, because the GCC pragmas
   work by disabling the diagnostic for a range of source locations
   and do not work when all the pragmas and the affected code are in a
   single macro expansion.  */

/* Push diagnostic state.  */
#define DIAG_PUSH_NEEDS_COMMENT _Pragma ("GCC diagnostic push")

/* Pop diagnostic state.  */
#define DIAG_POP_NEEDS_COMMENT _Pragma ("GCC diagnostic pop")

#define _DIAG_STR1(s) #s
#define _DIAG_STR(s) _DIAG_STR1(s)

/* Ignore the diagnostic OPTION.  VERSION is the most recent GCC
   version for which the diagnostic has been confirmed to appear in
   the absence of the pragma (in the form MAJOR.MINOR for GCC 4.x,
   just MAJOR for GCC 5 and later).  Uses of this pragma should be
   reviewed when the GCC version given is no longer supported for
   building glibc; the version number should always be on the same
   source line as the macro name, so such uses can be found with grep.
   Uses should come with a comment giving more details of the
   diagnostic, and an architecture on which it is seen if possibly
   optimization-related and not in architecture-specific code.  This
   macro should only be used if the diagnostic seems hard to fix (for
   example, optimization-related false positives).  */
#define DIAG_IGNORE_NEEDS_COMMENT(version, option)     \
  _Pragma (_DIAG_STR (GCC diagnostic ignored option))

/* Similar to DIAG_IGNORE_NEEDS_COMMENT the following macro ignores the
   diagnostic OPTION but only if optimizations for size are enabled.
   This is required because different warnings may be generated for
   different optimization levels.  For example a key piece of code may
   only generate a warning when compiled at -Os, but at -O2 you could
   still want the warning to be enabled to catch errors.  In this case
   you would use DIAG_IGNORE_Os_NEEDS_COMMENT to disable the warning
   only for -Os.  */
#ifdef __OPTIMIZE_SIZE__
# define DIAG_IGNORE_Os_NEEDS_COMMENT(version, option) \
  _Pragma (_DIAG_STR (GCC diagnostic ignored option))
#else
# define DIAG_IGNORE_Os_NEEDS_COMMENT(version, option)
#endif

#endif /* libc-diag.h */
