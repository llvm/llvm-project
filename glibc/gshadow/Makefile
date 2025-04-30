# Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
#	Makefile for gshadow.
#
subdir	:= gshadow

include ../Makeconfig

headers		= gshadow.h
routines	= getsgent getsgnam sgetsgent fgetsgent putsgent \
		  getsgent_r getsgnam_r sgetsgent_r fgetsgent_r

tests = tst-gshadow tst-putsgent tst-fgetsgent_r

CFLAGS-getsgent_r.c += -fexceptions
CFLAGS-getsgent.c += -fexceptions
CFLAGS-fgetsgent.c += -fexceptions
CFLAGS-fgetsgent_r.c += -fexceptions $(libio-mtsafe)
CFLAGS-putsgent.c += -fexceptions $(libio-mtsafe)
CFLAGS-getsgnam.c += -fexceptions
CFLAGS-getsgnam_r.c += -fexceptions

include ../Rules
