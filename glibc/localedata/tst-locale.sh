#!/bin/sh
# Testing the implementation of localedata.
# Copyright (C) 1998-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Andreas Jaeger, <aj@arthur.rhein-neckar.de>, 1998.

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

set -e

common_objpfx=$1; shift
localedef_before_env=$1; shift
run_program_env=$1; shift
localedef_after_env=$1; shift

test_locale ()
{
    charmap=$1
    input=$2
    out=$3
    rep=$4
    if test $rep; then
      rep="--repertoire-map $rep"
    fi
    # We expect the test locales to fail with warnings, they are mostly
    # incomplete and used for testing purposes, but that is OK.
    ret=0
    ${localedef_before_env} \
    ${run_program_env} \
    I18NPATH=. \
    ${localedef_after_env} --quiet -c -f $charmap -i $input \
      ${rep} ${common_objpfx}localedata/$out || ret=$?
    # Any error greater than one means we ran into an implementation
    # defined limit or saw an error that caused the output not to
    # be written, or lastly saw a fatal error that terminated
    # localedef.
    if [ $ret -gt 1 ]; then
	echo "Charmap: \"${charmap}\" Inputfile: \"${input}\"" \
	     "Outputdir: \"${out}\" failed"
	exit 1
    else
	echo -n "locale $out generated succesfully"
        if [ $ret -eq 1 ]; then
	    echo " (with warnings)"
        else
	    echo " (without warnings)"
        fi
    fi
}

test_locale IBM437 de_DE de_DE.437
test_locale tests/test1.cm tests/test1.def test1
test_locale tests/test2.cm tests/test2.def test2
test_locale tests/test3.cm tests/test3.def test3
test_locale tests/test4.cm tests/test4.def test4
test_locale tests/test5.cm tests/test5.def test5 tests/test5.ds
test_locale tests/test6.cm tests/test6.def test6 tests/test6.ds
test_locale tests/test7.cm tests/test7.def test7

exit 0

# Local Variables:
#  mode:shell-script
# End:
