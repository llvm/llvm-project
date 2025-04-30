/* Test TLS allocation with an interposed malloc.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* Reuse the test.  */
#define STACK_SIZE_MB 5
#include "tst-tls3.c"

/* Increase the thread stack size to 10 MiB, so that some thread
   stacks are actually freed.  (The stack cache size is currently
   hard-wired to 40 MiB in allocatestack.c.)  */
static long stack_size_in_mb = 10;

#include <sys/mman.h>

#define INTERPOSE_THREADS 1
#include "../malloc/tst-interpose-aux.c"
