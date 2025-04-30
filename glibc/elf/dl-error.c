/* Error handling for runtime dynamic linker, full version.
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

/* This implementation lives in libc.so because it uses thread-local
   data, which is not available in ld.so.  It interposes the version
   in dl-error-minimal.c after ld.so bootstrap.

   The signal/catch mechanism is used by the audit framework, which
   means that even in ld.so, not all errors are fatal.  */

#define DL_ERROR_BOOTSTRAP 0
#include "dl-error-skeleton.c"
