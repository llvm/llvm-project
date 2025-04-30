#!/bin/bash
# Run a testcase on a remote system, via ssh.
# Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

# usage: cross-test-ssh.sh [--ssh SSH] HOST COMMAND ...
# Run with --help flag to get more detailed help.

progname="$(basename $0)"

usage="usage: ${progname} [--ssh SSH] [--allow-time-setting] HOST COMMAND ..."
help="Run a glibc test COMMAND on the remote machine HOST, via ssh,
preserving the current working directory, and respecting quoting.

If the '--ssh SSH' flag is present, use SSH as the SSH command,
instead of ordinary 'ssh'.

If the '--timeoutfactor FACTOR' flag is present, set TIMEOUTFACTOR on
the remote machine to the specified FACTOR.

If the '--allow-time-setting' flag is present, set
GLIBC_TEST_ALLOW_TIME_SETTING on the remote machine to indicate that
time can be safely adjusted (e.g. on a virtual machine).

To use this to run glibc tests, invoke the tests as follows:

  $ make test-wrapper='ABSPATH/cross-test-ssh.sh HOST' tests

where ABSPATH is the absolute path to this script, and HOST is the
name of the machine to connect to via ssh.

If you need to connect to the test machine as a different user, you
may specify that just as you would to SSH:

  $ make test-wrapper='ABSPATH/cross-test-ssh.sh USER@HOST' tests

Naturally, the remote user must have an appropriate public key, and
you will want to ensure that SSH does not prompt interactively for a
password on each connection.

HOST and the build machines (on which 'make check' is being run) must
share a filesystem; all files needed by the tests must be visible at
the same paths on both machines.

${progname} runs COMMAND in the same directory on the HOST that
${progname} itself is run in on the build machine.

The command and arguments are passed to the remote host in a way that
avoids any further shell substitution or expansion, on the assumption
that the shell on the build machine has already done them
appropriately."

ssh='ssh'
timeoutfactor=$TIMEOUTFACTOR
while [ $# -gt 0 ]; do
  case "$1" in

    "--ssh")
      shift
      if [ $# -lt 1 ]; then
        break
      fi
      ssh="$1"
      ;;

    "--timeoutfactor")
      shift
      if [ $# -lt 1 ]; then
        break
      fi
      timeoutfactor="$1"
      ;;

    "--allow-time-setting")
      settimeallowed="1"
      ;;

    "--help")
      echo "$usage"
      echo "$help"
      exit 0
      ;;

    *)
      break
      ;;
  esac
  shift
done

if [ $# -lt 1 ]; then
  echo "$usage" >&2
  echo "Type '${progname} --help' for more detailed help." >&2
  exit 1
fi

host="$1"; shift

# Print the sequence of arguments as strings properly quoted for the
# Bourne shell, separated by spaces.
bourne_quote ()
{
  local arg qarg
  for arg in "$@"; do
    qarg=${arg//\'/\'\\\'\'}
    echo -n "'$qarg' "
  done
}

# Transform the current argument list into a properly quoted Bourne shell
# command string.
command="$(bourne_quote "$@")"

# Add command to set the current directory.
command="cd $(bourne_quote "$PWD")
${command}"

# Add command to set the timeout factor, if required.
if [ "$timeoutfactor" ]; then
  command="export TIMEOUTFACTOR=$(bourne_quote "$timeoutfactor")
${command}"
fi

# Add command to set the info that time on target can be adjusted,
# if required.
# Serialize execution of this script on target to prevent from unintended
# change of target time.
FLOCK_PATH="${FLOCK_PATH:-/var/lock/clock_settime}"
FLOCK_TIMEOUT="${FLOCK_TIMEOUT:-20}"
FLOCK_FD="${FLOCK_FD:-99}"
if [ "$settimeallowed" ]; then
  command="exec ${FLOCK_FD}<>${FLOCK_PATH}
flock -w ${FLOCK_TIMEOUT} ${FLOCK_FD}
if [ $? -ne 0 ]; then exit 1; fi
export GLIBC_TEST_ALLOW_TIME_SETTING=1
${command}"
fi

# HOST's sshd simply concatenates its arguments with spaces and
# passes them to some shell.  We want to force the use of /bin/sh,
# so we need to re-quote the whole command to ensure it appears as
# the sole argument of the '-c' option.
full_command="$(bourne_quote "${command}")"
$ssh "$host" /bin/sh -c "$full_command"
