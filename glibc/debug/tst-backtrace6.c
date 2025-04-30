/* Test backtrace and backtrace_symbols for signal frames, where a
   system call was interrupted by a signal.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <signal.h>

#ifdef SA_SIGINFO
# define SIGACTION_FLAGS SA_SIGINFO
# include <debug/tst-backtrace5.c>
#else
# define TEST_FUNCTION  0
# include "../test-skeleton.c"
#endif
