/* Test case for BZ #16634.  Non-PIE version.

   Verify that incorrectly dlopen()ing an executable without
   __RTLD_OPENEXEC does not cause assertion in ld.so, and that it
   actually results in an error.

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

#define TST_DLOPEN_TLSMODID_PATH "tst-dlopen-self"
#include "tst-dlopen-tlsmodid.h"
