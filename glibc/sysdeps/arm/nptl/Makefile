# Copyright (C) 2005-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
#
# The GNU C Library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# The GNU C Library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with the GNU C Library.  If not, see
# <https://www.gnu.org/licenses/>.

ifeq ($(subdir),csu)
gen-as-const-headers += tcb-offsets.sym
endif

ifeq ($(subdir),nptl)
# This test relies on compiling part of the binary with EH information,
# part without, and unwinding through.  The .ARM.exidx tables have
# start addresses for EH regions, but no end addresses.  Every
# region an exception needs to propogate through must have unwind
# information, or a previous function's unwind table may be used
# by mistake.
tests := $(filter-out tst-cleanupx4,$(tests))
endif
