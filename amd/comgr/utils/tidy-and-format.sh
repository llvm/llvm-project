#/bin/bash

set -euo pipefail

if ! test -f ../../../build/bin/clang-format; then
  printf "error: could not find clang-format in llvm-project/build/bin directory\n" >&2
  exit 1
fi

cd "$(git rev-parse --show-toplevel)/amd/comgr"

if [ ! -e compile_commands.json ]; then
  printf "error: compile_commands.json database missing\n" >&2
  printf " hint: enable with -DCMAKE_EXPORT_COMPILE_COMMANDS=On and then symlink into the amd/comgr directory:\n" >&2
  printf "  amd/comgr/build$ cmake ... -DCMAKE_EXPORT_COMPILE_COMMANDS=On ... && make && cd ..\n" >&2
  printf "  amd/comgr$ ln -s build/compile_commands.json .\n" >&2
  exit 1
fi

../../clang-tools-extra/clang-tidy/tool/run-clang-tidy.py -fix

# FIXME: Drive this off of compile_commands.json
find src/ test/ -type f -regex '.*\.\(c\|cpp\|h\|hpp\|cl\)$' -print0 \
  | xargs -0 ../../build/bin/clang-format -i
