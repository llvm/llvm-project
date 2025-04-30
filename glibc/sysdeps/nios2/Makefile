# Copyright (C) 1993-2021 Free Software Foundation, Inc.
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
# License along with the GNU C Library.  If not, see
# <https://www.gnu.org/licenses/>.

pic-ccflag = -fpic

# We don't support long doubles as a distinct type.  We don't need to set
# this variable; it's here mostly for documentational purposes.
long-double-fcts = no

ifeq ($(subdir),soft-fp)
sysdep_routines += $(gcc-single-routines) $(gcc-double-routines)
endif

ifeq ($(subdir),csu)
gen-as-const-headers += tcb-offsets.sym
endif
