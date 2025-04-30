#! /bin/sh
# Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

# For each installed header, confirm that it's possible to compile a
# file that includes that header and does nothing else, in several
# different compilation modes.

# These compilation switches assume GCC or compatible, which is probably
# fine since we also assume that when _building_ glibc.
c_modes="-std=c89 -std=gnu89 -std=c11 -std=gnu11"
cxx_modes="-std=c++98 -std=gnu++98 -std=c++11 -std=gnu++11"

# An exhaustive test of feature selection macros would take far too long.
# These are probably the most commonly used three.
lib_modes="-D_DEFAULT_SOURCE=1 -D_GNU_SOURCE=1 -D_XOPEN_SOURCE=700"

if [ $# -lt 3 ]; then
    echo "usage: $0 c|c++ \"compile command\" header header header..." >&2
    exit 2
fi
case "$1" in
    (c)
        lang_modes="$c_modes"
        cih_test_c=$(mktemp ${TMPDIR-/tmp}/cih_test_XXXXXX.c)
    ;;
    (c++)
        lang_modes="$cxx_modes"
        cih_test_c=$(mktemp ${TMPDIR-/tmp}/cih_test_XXXXXX.cc)
    ;;
    (*)
        echo "usage: $0 c|c++ \"compile command\" header header header..." >&2
        exit 2;;
esac
shift
cc_cmd="$1"
shift
trap "rm -f '$cih_test_c'" 0

failed=0
is_x86_64=unknown
for header in "$@"; do
    # Skip various headers for which this test gets a false failure.
    case "$header" in
        # bits/* are not meant to be included directly and usually #error
        # out if you try it.
        # regexp.h is a stub containing only an #error.
        # Sun RPC's .x files are traditionally installed in
        # $prefix/include/rpcsvc, but they are not C header files.
        (bits/* | regexp.h | rpcsvc/*.x)
            continue;;

        # All extant versions of sys/elf.h contain nothing more than an
        # exhortation (either a #warning or an #error) to use sys/procfs.h
        # instead, plus an inclusion of that header.
        (sys/elf.h)
            continue;;

        # Skip Fortran headers.
        (finclude/*)
            continue;;

        # sys/vm86.h is "unsupported on x86-64" and errors out on that target.
        (sys/vm86.h)
            case "$is_x86_64" in
                (yes) continue;;
                (no)  ;;
                (unknown)
                    cat >"$cih_test_c" <<EOF
#if defined __x86_64__ && __x86_64__
#error "is x86-64"
#endif
EOF
                    if $cc_cmd -fsyntax-only "$cih_test_c" > /dev/null 2>&1
                    then
                        is_x86_64=no
                    else
                        is_x86_64=yes
                        continue
                    fi
                ;;
            esac
            ;;
    esac

    echo :: "$header"
    for lang_mode in "" $lang_modes; do
        for lib_mode in "" $lib_modes; do
            echo :::: $lang_mode $lib_mode
            if [ -z "$lib_mode" ]; then
                expanded_lib_mode='/* default library mode */'
            else
                expanded_lib_mode=$(echo : $lib_mode | \
                    sed 's/^: -D/#define /; s/=/ /')
            fi
            cat >"$cih_test_c" <<EOF
/* These macros may have been defined on the command line.  They are
   inappropriate for this test.  */
#undef _LIBC
#undef _GNU_SOURCE
/* The library mode is selected here rather than on the command line to
   ensure that this selection wins. */
$expanded_lib_mode
#include <$header>
int avoid_empty_translation_unit;
EOF
            if $cc_cmd -finput-charset=ascii -fsyntax-only $lang_mode \
		       "$cih_test_c" 2>&1
            then :
            else failed=1
            fi
        done
    done
done
exit $failed
