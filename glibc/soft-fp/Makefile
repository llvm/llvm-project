# Copyright (C) 1997-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
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

#
#	Makefile for software floating-point routines
#
subdir	:= soft-fp

include ../Makeconfig

gcc-single-routines := negsf2 addsf3 subsf3 mulsf3 divsf3 eqsf2 \
	lesf2 gesf2 unordsf2 fixsfsi fixunssfsi floatsisf fixsfdi        \
	fixunssfdi floatdisf floatunsisf floatundisf

gcc-double-routines := negdf2 adddf3 subdf3 muldf3 divdf3 eqdf2 \
	ledf2 gedf2 unorddf2 fixdfsi fixunsdfsi floatsidf fixdfdi        \
	fixunsdfdi floatdidf extendsfdf2 truncdfsf2 floatunsidf \
	floatundidf

gcc-quad-routines := negtf2 addtf3 subtf3 multf3 divtf3 eqtf2 \
	letf2 getf2 unordtf2 fixtfsi fixunstfsi floatsitf fixtfdi      \
	fixunstfdi floatditf extendsftf2 trunctfsf2 extenddftf2 \
	trunctfdf2 floatunsitf floatunditf

include ../Rules
