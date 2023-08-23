#/bin/bash

set -euo pipefail

hash git find clang-format clang-tidy clang-apply-replacements

if ! hash run-clang-tidy.py 2>/dev/null; then
  # For some reason this doesn't get installed into the PATH on e.g. Ubuntu, so
  # let's see if it is in a known location rather than just fail.
  if [ -x /usr/local/share/clang/run-clang-tidy.py ]; then
    hash -p /usr/local/share/clang/run-clang-tidy.py run-clang-tidy.py
  else
    printf "error: could not find run-clang-tidy.py in PATH\n" >&2
    exit 1
  fi
fi

cd "$(git rev-parse --show-toplevel)/lib/comgr"

if [ ! -e compile_commands.json ]; then
  printf "error: compile_commands.json database missing\n" >&2
  printf " hint: enable with -DCMAKE_EXPORT_COMPILE_COMMANDS=On and then symlink into the lib/comgr directory:\n" >&2
  printf "  lib/comgr/release$ cmake ... -DCMAKE_EXPORT_COMPILE_COMMANDS=On ... && make && cd ..\n" >&2
  printf "  lib/comgr$ ln -s release/compile_commands.json .\n" >&2
  exit 1
fi

run-clang-tidy.py -fix

# FIXME: Drive this off of compile_commands.json
find src/ test/ -type f -regex '.*\.\(c\|cpp\|h\|hpp\|cl\)$' -print0 \
  | xargs -0 clang-format -i
