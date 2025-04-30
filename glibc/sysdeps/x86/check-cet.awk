# Verify that all shared objects contain the CET property.
# Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

# This awk script expects to get command-line files that are each
# the output of 'readelf -n' on a single shared object.
# It exits successfully (0) if all of them contained the CET property.
# It fails (1) if any didn't contain the CET property
# It fails (2) if the input did not take the expected form.

BEGIN { result = cet = sanity = 0 }

function check_one(name) {
  if (!sanity) {
    print name ": *** input did not look like readelf -n output";
    result = 2;
  } else if (cet) {
    print name ": OK";
  } else {
    print name ": *** no CET property found";
    result = result ? result : 1;
  }

  cet = sanity = 0;
}

FILENAME != lastfile {
  if (lastfile)
    check_one(lastfile);
  lastfile = FILENAME;
}

index ($0, "Displaying notes") != 0 { sanity = 1 }
index ($0, "IBT") != 0 && index ($0, "SHSTK") != 0 { cet = 1 }

END {
  check_one(lastfile);
  exit(result);
}
