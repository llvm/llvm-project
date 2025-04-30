/* Runtime architecture check for math tests.
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

/* These macros used for architecture check in math tests runtime.
   INIT_ARCH_EXT should set up for example some global variable which is
   checked by CHECK_ARCH_EXT which produces return from individual test to
   prevent run on hardware not supported by tested function implementation. */
#define INIT_ARCH_EXT
#define CHECK_ARCH_EXT
