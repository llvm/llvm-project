#!/bin/bash
# Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

# We should be able to find the translation right at the beginning.
TEXTDOMAIN=libc
TEXTDOMAINDIR=@TEXTDOMAINDIR@

unset SOTRUSS_FROMLIST
unset SOTRUSS_TOLIST
unset SOTRUSS_OUTNAME
unset SOTRUSS_EXIT
unset SOTRUSS_NOINDENT
SOTRUSS_WHICH=$$
lib='@PREFIX@/$LIB/audit/sotruss-lib.so'

do_help() {
  echo $"Usage: sotruss [OPTION...] [--] EXECUTABLE [EXECUTABLE-OPTION...]
  -F, --from FROMLIST     Trace calls from objects on FROMLIST
  -T, --to TOLIST         Trace calls to objects on TOLIST

  -e, --exit              Also show exits from the function calls
  -f, --follow            Trace child processes
  -o, --output FILENAME   Write output to FILENAME (or FILENAME.$PID in case
			  -f is also used) instead of standard error

  -?, --help              Give this help list
      --usage             Give a short usage message
      --version           Print program version"

  echo
  printf $"Mandatory arguments to long options are also mandatory for any corresponding\nshort options.\n"
  echo

  printf $"For bug reporting instructions, please see:\\n%s.\\n" \
    "@REPORT_BUGS_TO@"
  exit 0
}

do_missing_arg() {
  printf >&2 $"%s: option requires an argument -- '%s'\n" sotruss "$1"
  printf >&2 $"Try \`%s --help' or \`%s --usage' for more information.\n" sotruss sotruss
  exit 1
}

do_ambiguous() {
  printf >&2 $"%s: option is ambiguous; possibilities:"
  while test $# -gt 0; do
    printf >&2 " '%s'" $1
    shift
  done
  printf >&2 "\n"
  printf >&2 $"Try \`%s --help' or \`%s --usage' for more information.\n" sotruss sotruss
  exit 1
}

while test $# -gt 0; do
  case "$1" in
  --v | --ve | --ver | --vers | --versi | --versio | --version)
    echo "sotruss @PKGVERSION@@VERSION@"
    printf $"Copyright (C) %s Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
" "2021"
    printf $"Written by %s.\n" "Ulrich Drepper"
    exit 0
    ;;
  -\? | --h | --he | --hel | --help)
    do_help
    ;;
  --u | --us | --usa | --usag | --usage)
    printf $"Usage: %s [-ef] [-F FROMLIST] [-o FILENAME] [-T TOLIST] [--exit]
	    [--follow] [--from FROMLIST] [--output FILENAME] [--to TOLIST]
	    [--help] [--usage] [--version] [--]
	    EXECUTABLE [EXECUTABLE-OPTION...]\n" sotruss
    exit 0
    ;;
  -F | --fr | --fro | --from)
    if test $# -eq 1; then
      do_missing_arg "$1"
    fi
    shift
    SOTRUSS_FROMLIST="$1"
    ;;
  -T | --t | --to)
    if test $# -eq 1; then
      do_missing_arg "$1"
    fi
    shift
    SOTRUSS_TOLIST="$1"
    ;;
  -o | --o | --ou | --out | --outp | --outpu | --output)
    if test $# -eq 1; then
      do_missing_arg "$1"
    fi
    shift
    SOTRUSS_OUTNAME="$1"
    ;;
  -f | --fo | --fol | --foll | --follo | --follow)
    unset SOTRUSS_WHICH
    ;;
  -l | --l | --li | --lib)
    if test $# -eq 1; then
      do_missing_arg "$1"
    fi
    shift
    lib="$1"
    ;;
  -e | --e | --ex | --exi | --exit)
    SOTRUSS_EXIT=1
    ;;
  --f)
    do_ambiguous '--from' '--follow'
    ;;
  --)
    shift
    break
    ;;
  -*)
    printf >&2 $"%s: unrecognized option '%c%s'\n" sotruss '-' ${1#-}
    printf >&2 $"Try \`%s --help' or \`%s --usage' for more information.\n" sotruss sotruss
    exit 1
    ;;
  *)
    break
    ;;
  esac
  shift
done

export SOTRUSS_FROMLIST
export SOTRUSS_TOLIST
export SOTRUSS_OUTNAME
export SOTRUSS_WHICH
export SOTRUSS_EXIT
export LD_AUDIT="$lib"

exec "$@"
