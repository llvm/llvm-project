#!/bin/bash
# Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

# To test BZ #18872, we need a printf() with 10K arguments.
# Such a printf could be generated with non-trivial macro
# application, but it's simpler to generate the test source
# via this script.

n_args=10000

cat <<'EOF'
#include <stdio.h>
#include <mcheck.h>

/*
  Compile do_test without optimization: GCC 4.9/5.0/6.0 takes a long time
  to build this source. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=67396  */

__attribute__ ((optimize ("-O0")))
int do_test (void)
{
    mtrace ();
    printf (
EOF

for ((j = 0; j < $n_args / 10; j++)); do
  for ((k = 0; k < 10; k++)); do
    printf '"%%%d$s" ' $((10 * $j + $k + 1))
  done
  printf "\n"
done

printf '"%%%d$s",\n' $(($n_args + 1))

for ((j = 0; j < $n_args / 10; j++)); do
  for ((k = 0; k < 10; k++)); do
    printf '"a", '
  done
  printf "  /* %4d */\n" $((10 * $j + $k))
done

printf '"\\n");'


cat <<'EOF'

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

EOF
