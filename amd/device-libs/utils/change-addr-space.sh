#!/bin/sh

# Usage:
# utils/change-addr-space.sh   : apply utils/add_amdgiz.sed
#                                adopt generic address space is address space 0
# utils/change-addr-space.sh x : apply utils/remove_amdgiz.sed
#                                adopt generic address space is address space 4

if [ $# -lt 1 ]; then
  find . -name "*.ll" | xargs sed -i -f "utils/add_amdgiz.sed"
else
  find . -name "*.ll" | xargs sed -i -f "utils/remove_amdgiz.sed"
fi

