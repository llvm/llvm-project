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
#	Sub-makefile for grp portion of the library.
#
subdir	:= grp

include ../Makeconfig

headers := grp.h

routines := fgetgrent initgroups setgroups \
	    getgrent getgrgid getgrnam putgrent \
	    getgrent_r getgrgid_r getgrnam_r fgetgrent_r \
	    grp-merge

tests := testgrp tst-putgrent

tests-container = \
	tst-initgroups1 \
	tst-initgroups2

ifeq (yes,$(build-shared))
test-srcs :=  tst_fgetgrent
ifeq ($(run-built-tests),yes)
tests-special += $(objpfx)tst_fgetgrent.out
endif
endif


include ../Rules

ifeq ($(have-thread-library),yes)

CFLAGS-getgrgid_r.c += -fexceptions
CFLAGS-getgrnam_r.c += -fexceptions
CFLAGS-getgrent_r.c += -fexceptions
CFLAGS-getgrent.c += -fexceptions
CFLAGS-fgetgrent.c += -fexceptions
CFLAGS-fgetgrent_r.c += -fexceptions $(libio-mtsafe)
CFLAGS-putgrent.c += -fexceptions $(libio-mtsafe)
CFLAGS-initgroups.c += -fexceptions
CFLAGS-getgrgid.c += -fexceptions

endif

ifeq ($(run-built-tests),yes)
# tst_fgetgrent currently only works with shared libraries
ifeq (yes,$(build-shared))
$(objpfx)tst_fgetgrent.out: tst_fgetgrent.sh $(objpfx)tst_fgetgrent
	$(SHELL) $< $(common-objpfx) '$(test-program-prefix)'; \
	$(evaluate-test)

endif
endif
