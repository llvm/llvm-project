#  Makefile for SPARC 64-bit.
#  Copyright (C) 1997-2021 Free Software Foundation, Inc.
#  This file is part of the GNU C Library.
#  Contributed by Jakub Jelinek (jj@ultra.linux.cz).
#

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

sysdep-CFLAGS += -Wa,-Av9a -mvis

ifeq ($(subdir),string)
sysdep_routines += align-cpy
endif

ASFLAGS-.o += -Wa,-Av9d
ASFLAGS-.os += -Wa,-Av9d
ASFLAGS-.op += -Wa,-Av9d
ASFLAGS-.oS += -Wa,-Av9d

ifeq ($(subdir),soft-fp)
sparc64-quad-routines := qp_add qp_cmp qp_cmpe qp_div qp_dtoq qp_feq qp_fge \
	qp_fgt qp_fle qp_flt qp_fne qp_itoq qp_mul qp_neg qp_qtod qp_qtoi   \
	qp_qtos qp_qtoui qp_qtoux qp_qtox qp_sqrt qp_stoq qp_sub qp_uitoq   \
	qp_uxtoq qp_xtoq qp_util
sysdep_routines += $(sparc64-quad-routines)
endif

ifeq ($(subdir),math)
CPPFLAGS += -I../soft-fp/
endif
