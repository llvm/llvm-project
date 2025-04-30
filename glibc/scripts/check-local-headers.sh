#!/bin/bash
# Copyright (C) 2005-2021 Free Software Foundation, Inc.
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
#
includedir="$1"
objpfx="$2"

# To avoid long paths.
cd "$objpfx"

# OK if *.os is missing.
shopt -s nullglob

# Search all dependency files for file names in the include directory.
# There are a few system headers we are known to use.
# These include Linux kernel headers (asm*, arch, and linux),
# and Mach kernel headers (mach).
exec ${AWK} -v includedir="$includedir" '
BEGIN {
  status = 0
  exclude = "^" includedir \
    "/(.*-.*-.*/|.*-.*/|)(asm[-/]|arch|linux/|selinux/|mach/|mach_debug/|device/|hurd/(((hurd|ioctl)_types|paths)\\.h|ioctls\\.defs|ihash\\.h|version\\.h)|gd|nss3/|nspr4?/|c\\+\\+/|sys/(capability|sdt(|-config))\\.h|libaudit\\.h)"
}
/^[^ ]/ && $1 ~ /.*:/ { obj = $1 }
{
  for (i = 1; i <= NF; ++i) {
    if ($i ~ ("^" includedir) && $i !~ exclude) {
      print "***", obj, "uses", $i
      status = 1
    }
  }
}
END { exit status }' */*.{o,os,oS}.d
