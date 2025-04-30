/* The uc_sigmask on IA64 has the wrong type and this needs fixing,
   but until that change is evaluated, we fix this here with a cast.
   See https://sourceware.org/bugzilla/show_bug.cgi?id=21634
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#undef sigismember
#define sigismember(set, signo) sigismember ((const sigset_t *) (set), (signo))

#include <stdlib/tst-setcontext4.c>
