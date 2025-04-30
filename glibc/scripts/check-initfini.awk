# Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

# This awk script expects to get command-line files that are each
# the output of 'readelf -W --dyn-syms' on a single shared object.
# It exits successfully (0) if none contained _init nor _fini in dynamic
# symbol table.
# It fails (1) if any did contain _init or _fini in dynamic symbol table.
# It fails (2) if the input did not take the expected form.

BEGIN { result = _init = _fini = sanity = 0 }

function check_one(name) {
  if (!sanity) {
    print name ": *** input did not look like readelf -d output";
    result = 2;
  } else {
    ok = 1;
    if (_init) {
      print name ": *** _init is in dynamic symbol table";
      result = result ? result : 1;
      ok = 0;
    }
    if (_fini) {
      print name ": *** _fini is in dynamic symbol table";
      result = result ? result : 1;
      ok = 0;
    }
    if (ok)
      print name ": OK";
  }

  _init = _fini = sanity = 0
}

FILENAME != lastfile {
  if (lastfile)
    check_one(lastfile);
  lastfile = FILENAME;
}

$1 == "Symbol" && $2 == "table" && $3 == "'.dynsym'" { sanity = 1 }
$8 == "_init" { _init = 1 }
$8 == "_fini" { _fini = 1 }

END {
  check_one(lastfile);
  exit(result);
}
