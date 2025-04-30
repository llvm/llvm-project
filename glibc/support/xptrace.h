/* Support functions handling ptrace_scope.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef SUPPORT_PTRACE_H
#define SUPPORT_PTRACE_H

#include <sys/cdefs.h>

__BEGIN_DECLS

/* Return the current YAMA mode set on the machine (0 to 3) or -1
   if YAMA is not supported.  */
int support_ptrace_scope (void);

__END_DECLS

#endif
