/* Default definition for ARGP_PROGRAM_VERSION.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Miles Bader <miles@gnu.ai.mit.edu>.

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

/* If set by the user program to a non-zero value, then a default option
   --version is added (unless the ARGP_NO_HELP flag is used), which will
   print this this string followed by a newline and exit (unless the
   ARGP_NO_EXIT flag is used).  Overridden by ARGP_PROGRAM_VERSION_HOOK.  */
const char *argp_program_version;
