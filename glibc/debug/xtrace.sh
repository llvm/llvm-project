#!/bin/bash
# Copyright (C) 1999-2021 Free Software Foundation, Inc.
# This file is part of the GNU C Library.
# Contributed by Ulrich Drepper <drepper@gnu.org>, 1999.

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

pcprofileso='@SLIBDIR@/libpcprofile.so'
pcprofiledump='@BINDIR@/pcprofiledump'
TEXTDOMAIN=libc

# Print usage message.
do_usage() {
  printf $"Usage: xtrace [OPTION]... PROGRAM [PROGRAMOPTION]...\n"
  exit 0
}

# Refer to --help option.
help_info() {
  printf >&2 $"Try \`%s --help' or \`%s --usage' for more information.\n" xtrace xtrace
  exit 1
}

# Message for missing argument.
do_missing_arg() {
  printf >&2 $"%s: option '%s' requires an argument.\n" xtrace "$1"
  help_info
}

# Print help message
do_help() {
  printf $"Usage: xtrace [OPTION]... PROGRAM [PROGRAMOPTION]...\n"
  printf $"Trace execution of program by printing currently executed function.

     --data=FILE          Don't run the program, just print the data from FILE.

   -?,--help              Print this help and exit
      --usage             Give a short usage message
   -V,--version           Print version information and exit

Mandatory arguments to long options are also mandatory for any corresponding
short options.

"
  printf $"For bug reporting instructions, please see:\\n%s.\\n" \
    "@REPORT_BUGS_TO@"
  exit 0
}

do_version() {
  echo 'xtrace @PKGVERSION@@VERSION@'
  printf $"Copyright (C) %s Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
" "2021"
  printf $"Written by %s.
" "Ulrich Drepper"
  exit 0
}

# Print out function name, file, and line number in a nicely formatted way.
format_line() {
  fct=$1
  file=${2%%:*}
  line=${2##*:}
  width=$(expr $COLUMNS - 30)
  filelen=$(expr length $file)
  if test "$filelen" -gt "$width"; then
    rwidth=$(expr $width - 3)
    file="...$(expr substr $file $(expr 1 + $filelen - $rwidth) $rwidth)"
  fi
  printf '%-20s  %-*s  %6s\n' $fct $width $file $line
}


# If the variable COLUMNS is not set do this now.
COLUMNS=${COLUMNS:-80}

# If `TERMINAL_PROG' is not set, set it to `xterm'.
TERMINAL_PROG=${TERMINAL_PROG:-xterm}

# The data file to process, if any.
data=

# Process arguments.  But stop as soon as the program name is found.
while test $# -gt 0; do
  case "$1" in
  --d | --da | --dat | --data)
    if test $# -eq 1; then
      do_missing_arg $1
    fi
    shift
    data="$1"
    ;;
  --d=* | --da=* | --dat=* | --data=*)
    data=${1##*=}
    ;;
  -\? | --h | --he | --hel | --help)
    do_help
    ;;
  -V | --v | --ve | --ver | --vers | --versi | --versio | --version)
    do_version
    ;;
  --u | --us | --usa | --usag | --usage)
    do_usage
    ;;
  --)
    # Stop processing arguments.
    shift
    break
    ;;
  --*)
    printf >&2 $"xtrace: unrecognized option \`$1'\n"
    help_info
    ;;
  *)
    # Unknown option.  This means the rest is the program name and parameters.
    break
    ;;
  esac
  shift
done

# See whether any arguments are left.
if test $# -eq 0; then
  printf >&2 $"No program name given\n"
  help_info
fi

# Determine the program name and check whether it exists.
program=$1
shift
if test ! -f "$program"; then
  printf >&2 $"executable \`$program' not found\n"
  help_info
fi
if test ! -x "$program"; then
  printf >&2 $"\`$program' is no executable\n"
  help_info
fi

# We have two modes.  If a data file is given simply print the included data.
printf "%-20s  %-*s  %6s\n" Function $(expr $COLUMNS - 30) File Line
for i in $(seq 1 $COLUMNS); do printf -; done; printf '\n'
if test -n "$data"; then
  $pcprofiledump "$data" |
  sed 's/this = \([^,]*\).*/\1/' |
  addr2line -fC -e "$program" |
  while read fct; do
    read file
    if test "$fct" != '??' -a "$file" != '??:0'; then
      format_line "$fct" "$file"
    fi
  done
else
  fifo=$(mktemp -ut xtrace.XXXXXX) || exit
  trap 'rm -f "$fifo"; exit 1' HUP INT QUIT TERM PIPE
  mkfifo -m 0600 $fifo || exit 1

  # Now start the program and let it write to the FIFO.
  $TERMINAL_PROG -T "xtrace - $program $*" -e /bin/sh -c "LD_PRELOAD=$pcprofileso PCPROFILE_OUTPUT=$fifo $program $*; read < $fifo" &
  termpid=$!
  $pcprofiledump -u "$fifo" |
  while read line; do
     echo "$line" |
     sed 's/this = \([^,]*\).*/\1/' |
     addr2line -fC -e "$program"
  done |
  while read fct; do
    read file
    if test "$fct" != '??' -a "$file" != '??:0'; then
      format_line "$fct" "$file"
    fi
  done
  read -p "Press return here to close $TERMINAL_PROG($program)."
  echo > "$fifo"
  rm "$fifo"
fi

exit 0
# Local Variables:
#  mode:ksh
# End:
