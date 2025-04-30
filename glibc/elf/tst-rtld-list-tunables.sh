#!/bin/sh
# Test for --list-tunables option ld.so.
# Copyright (C) 2021 Free Software Foundation, Inc.
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
# License along with the GNU C Library; if not, see
# <https://www.gnu.org/licenses/>.

set -e

rtld=$1
test_wrapper_env=$2
run_program_env=$3

LC_ALL=C
export LC_ALL

# Unset tunables and their aliases.
GLIBC_TUNABLES=
MALLOC_ARENA_MAX=
MALLOC_ARENA_TEST=
MALLOC_CHECK_=
MALLOC_MMAP_MAX_=
MALLOC_MMAP_THRESHOLD_=
MALLOC_PERTURB_=
MALLOC_TOP_PAD_=
MALLOC_TRIM_THRESHOLD_=

${test_wrapper_env} \
${run_program_env} \
$rtld --list-tunables \
| sort -u \
| egrep "(rtld|malloc)" \
| sed -e "s/0xf\+/0x[f]+/"
