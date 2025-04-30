#!/bin/bash
# Create a patch which backports the support/ subdirectory.
# Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

# This script does not backport the Makefile tweaks outside the
# support/ directory (which need to be backported separately), or the
# changes to test-skeleton.c (which should not be backported).

set -e

export LC_ALL=C
export GIT_CONFIG=/dev/null
export GTT_CONFIG_NOSYSTEM=0
export GIT_PAGER=

usage () {
    cat >&2 <<EOF
usage: $0 {patch|commit}
EOF
    exit 1
}

if test $# -ne 1 ; then
    usage
fi

command="$1"

case "$command" in
    patch|commit)
    ;;
    *)
	usage
	;;
esac

# The upstream branch to work on.
branch=origin/master

# The commit which added the support/ directory.
initial_commit=c23de0aacbeaa7a091609b35764bed931475a16d

# We backport the support directory and this script.  Directories need
# to end in a /.
patch_targets="support/ scripts/backport-support.sh"

latest_commit="$(git log --max-count=1 --pretty=format:%H "$branch" -- \
  $patch_targets)"

# Simplify the branch name somewhat for reporting.
branch_name="$(echo "$branch" | sed s,^origin/,,)"

command_patch () {
    cat <<EOF
This patch creates the contents of the support/ directory up to this
upstream commit on the $branch_name branch:

EOF
    git log --max-count=1 "$latest_commit"
    echo
    git diff "$initial_commit"^.."$latest_commit" $patch_targets
    echo "# Before applying the patch, run this command:" >&2
    echo "# rm -rf $patch_targets" >&2
}

command_commit () {
    git status --porcelain | while read line ; do
	echo "error: working copy is not clean, cannot commit" >&2
	exit 1
    done
    for path in $patch_targets; do
	echo "# Processing $path" >&2
	case "$path" in
	    [a-zA-Z0-9]*/)
		# Directory.
		git rm --cached --ignore-unmatch -r "$path"
		rm -rf "$path"
		git read-tree --prefix="$path" "$latest_commit":"$path"
		git checkout "$path"
		;;
	    *)
		# File.
		git show "$latest_commit":"$path" > "$path"
		git add "$path"
	esac
    done
    git commit -m "Synchronize support/ infrastructure with $branch_name

This commit updates the support/ subdirectory to
commit $latest_commit
on the $branch_name branch.
"
}

command_$command
