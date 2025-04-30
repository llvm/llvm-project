/* Constant values for the uname function to return.  Generic version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* This file exists so that it can be replaced by sysdeps variants.
   It must define these macros with string values:
        UNAME_SYSNAME
        UNAME_RELEASE
        UNAME_VERSION
        UNAME_MACHINE
   If there is no sysdeps file, this file will just proxy to the file
   created by posix/Makefile.  */

#include <config-name.h>
