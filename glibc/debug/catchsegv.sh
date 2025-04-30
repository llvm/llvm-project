#!/bin/sh
# Copyright (C) 1998-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

if test $# -eq 0; then
  echo "$0: missing program name" >&2
  echo "Try \`$0 --help' for more information." >&2
  exit 1
fi

prog="$1"
shift

if test $# -eq 0; then
  case "$prog" in
    --h | --he | --hel | --help)
      echo 'Usage: catchsegv PROGRAM ARGS...'
      echo '  --help      print this help, then exit'
      echo '  --version   print version number, then exit'
      echo 'For bug reporting instructions, please see:'
      cat <<\EOF
@REPORT_BUGS_TO@.
EOF
      exit 0
      ;;
    --v | --ve | --ver | --vers | --versi | --versio | --version)
      echo 'catchsegv @PKGVERSION@@VERSION@'
      echo 'Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
Written by Ulrich Drepper.'
      exit 0
      ;;
    *)
      ;;
  esac
fi

segv_output=`mktemp ${TMPDIR:-/tmp}/segv_output.XXXXXX` || exit

# Redirect stderr to avoid termination message from shell.
(exec 3>&2 2>/dev/null
LD_PRELOAD=${LD_PRELOAD:+${LD_PRELOAD}:}@SLIB@/libSegFault.so \
SEGFAULT_USE_ALTSTACK=1 \
SEGFAULT_OUTPUT_NAME=$segv_output \
"$prog" ${1+"$@"} 2>&3 3>&-)
exval=$?

# Check for output.  Even if the program terminated correctly it might
# be that a minor process (clone) failed.  Therefore we do not check the
# exit code.
if test -s "$segv_output"; then
  # The program caught a signal.  The output is in the file with the
  # name we have in SEGFAULT_OUTPUT_NAME.  In the output the names of
  # functions in shared objects are available, but names in the static
  # part of the program are not.  We use addr2line to get this information.
  case $prog in
  */*) ;;
  *)
    old_IFS=$IFS
    IFS=:
    for p in $PATH; do
      test -n "$p" || p=.
      if test -f "$p/$prog"; then
	prog=$p/$prog
      break
      fi
    done
    IFS=$old_IFS
    ;;
  esac
  sed '/Backtrace/q' "$segv_output"
  sed '1,/Backtrace/d' "$segv_output" |
  (while read line; do
     line=`echo $line | sed "s@^$prog\\(\\[.*\\)@\1@"`
     case "$line" in
       \[*) addr=`echo "$line" | sed 's/^\[\(.*\)\]$/\1/'`
	    complete=`addr2line -f -e "$prog" $addr 2>/dev/null`
	    if test $? -eq 0; then
	      echo "`echo "$complete"|sed 'N;s/\(.*\)\n\(.*\)/\2(\1)/;'`$line"
	    else
	      echo "$line"
	    fi
	    ;;
	 *) echo "$line"
	    ;;
     esac
   done)
fi
rm -f "$segv_output"

exit $exval
