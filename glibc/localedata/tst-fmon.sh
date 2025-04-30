#!/bin/sh
# Testing the implementation of strfmon(3).
# Copyright (C) 1996-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Jochen Hein <jochen.hein@delphi.central.de>, 1997.

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

common_objpfx=$1
run_program_prefix_before_env=$2
run_program_env=$3
run_program_prefix_after_env=$4
test_program_prefix=$5
datafile=$6

here=`pwd`

lang=`sed -e '/^#/d' -e '/^$/d' -e '/^C	/d' -e '/^tstfmon/d' -e 's/^\([^	]*\).*/\1/' $datafile | sort | uniq`

# Generate data files.
for cns in `cd ./tst-fmon-locales && ls tstfmon_*`; do
    ret=0
    cn=tst-fmon-locales/$cns
    fn=charmaps/ISO-8859-1
    # All of the test locales run with "USC " as their int_curr_symbol,
    # and the use of this generates a warning because it does not meet
    # the POSIX requirement that the name be an ISO 4217 compliant
    # country code e.g. USD.  Therefore we *expect* an exit code of 1.
    ${run_program_prefix_before_env} \
    ${run_program_env} \
    I18NPATH=. \
    ${run_program_prefix_after_env} ${common_objpfx}locale/localedef \
    --quiet -i $cn -f $fn ${common_objpfx}localedata/$cns || ret=$?
    if [ $ret -ne 1 ]; then
	echo "FAIL: Locale compilation for $cn failed (error $ret)."
	exit 1
    fi
done

# Run the tests.
errcode=0
# There's a TAB for IFS
while IFS="	" read locale format value expect; do
    case "$locale" in '#'*) continue ;; esac
    if [ -n "$format" ]; then
	ret=0
	expect=`echo "$expect" | sed 's/^\"\(.*\)\"$/\1/'`
	${test_program_prefix} ${common_objpfx}localedata/tst-fmon \
	"$locale" "$format" "$value" "$expect" < /dev/null || ret=$?
        if [ $ret -ne 0 ]; then
	    echo "FAIL: Locale $locale failed the test (error $ret)."
	    errcode=1
	fi
    fi
done < $datafile

exit $errcode
# Local Variables:
#  mode:shell-script
# End:
