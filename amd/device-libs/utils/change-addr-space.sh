#!/bin/sh

# Usage:
# utils/change-addr-space.sh src  : apply utils/add_amdgiz.sed
#                                adopt generic address space is address space 0
# utils/change-addr-space.sh src x : apply utils/remove_amdgiz.sed
#                                adopt generic address space is address space 4

if [ $# -lt 2 ]; then
  find . -name "*.ll" | xargs sed -i -f "$1/add_amdgiz.sed"
else
  find . -name "*.ll" | xargs sed -i -f "$1/remove_amdgiz.sed"
fi
