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

subdir := resource

include ../Makeconfig

headers	  := sys/resource.h bits/resource.h sys/vlimit.h	\
	     ulimit.h bits/types/struct_rusage.h

routines := getrlimit setrlimit getrlimit64 setrlimit64 getrusage ulimit      \
	    vlimit vtimes getpriority setpriority nice

tests = tst-getrlimit bug-ulimit1

include ../Rules
