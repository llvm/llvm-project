/* Determine DT_INIT/DT_FINI support in the dynamic loader.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

/* Legacy platforms define this to 1.  If 0, the dynamic linker
   ignores the DT_INIT and DT_FINI tags, and static binaries will not
   call the _init or _fini functions.  If 1, the old constructor
   mechanisms are used in addition to the initarray/finiarray
   support.  */
#define ELF_INITFINI 0
