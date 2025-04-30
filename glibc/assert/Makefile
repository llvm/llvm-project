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
#	Sub-makefile for assert portion of the library.
#
subdir	:= assert

include ../Makeconfig

headers	:= assert.h

routines := assert assert-perr __assert
tests := test-assert test-assert-perr tst-assert-c++ tst-assert-g++

ifeq ($(have-cxx-thread_local),yes)
CFLAGS-tst-assert-c++.o = -std=c++11
LDLIBS-tst-assert-c++ = -lstdc++
CFLAGS-tst-assert-g++.o = -std=gnu++11
LDLIBS-tst-assert-g++ = -lstdc++
else
tests-unsupported += tst-assert-c++ tst-assert-g++
endif

include ../Rules
