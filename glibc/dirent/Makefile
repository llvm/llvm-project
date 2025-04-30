# Copyright (C) 1991-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

#
#	Sub-makefile for dirent portion of the library.
#
subdir		:= dirent

include ../Makeconfig

headers		:= dirent.h bits/dirent.h bits/dirent_ext.h
routines	:= opendir closedir readdir readdir_r rewinddir \
		   seekdir telldir scandir alphasort versionsort \
		   getdents getdents64 dirfd readdir64 readdir64_r scandir64 \
		   alphasort64 versionsort64 fdopendir \
		   scandirat scandirat64 \
		   scandir-cancel scandir-tail scandir64-tail

tests	   := list tst-seekdir opendir-tst1 bug-readdir1 tst-fdopendir \
	      tst-fdopendir2 tst-scandir tst-scandir64

CFLAGS-scandir.c += $(uses-callbacks)
CFLAGS-scandir64.c += $(uses-callbacks)
CFLAGS-scandir-tail.c += $(uses-callbacks)
CFLAGS-scandir64-tail.c += $(uses-callbacks)

include ../Rules

opendir-tst1-ARGS = --test-dir=${common-objpfx}dirent
