# Copyright (C) 2000-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by David Huggins-Daines (dhd@debian.org)

# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.

# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library.  If not, see
# <https://www.gnu.org/licenses/>.

# We used to need this since the build process uses ld -r.  Now we use
# ld -r --unique=.text* which does more or less the same thing, but better.
# CFLAGS-.os += -ffunction-sections
LDFLAGS-c_pic.os += -Wl,--unique=.text*

ifeq ($(subdir),elf)
CFLAGS-rtld.c += -mdisable-fpregs
sysdep-dl-routines += dl-symaddr dl-fptr
# dl-fptr.c needs a complete rewrite to fix ISO C aliasing violations.
CFLAGS-dl-fptr.c = -Wno-error
endif

ifeq ($(subdir),csu)
ifeq (yes,$(build-shared))
# Compatibility
ifeq (yes,$(have-protected))
CPPFLAGS-libgcc-compat.c = -DHAVE_DOT_HIDDEN
endif
sysdep_routines += libgcc-compat
shared-only-routines += libgcc-compat
endif
endif

# We implement a 64-bit `long double'. The standard says we can do this.
# This means our `long double' and `double' are identical.
long-double-fcts = no
