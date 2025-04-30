#!/bin/sh
#
# List all the files under version control in the source tree.
#

case $# in
0) ;;
1) cd "$1" ;;
*) echo >&2 "Usage: $0 [top_srcdir]"; exit 2 ;;
esac

if [ -r .git/HEAD ]; then
  ${GIT:-git} ls-files
  exit 0
fi

echo >&2 'Cannot list sources without some version control system in use.'
exit 1
