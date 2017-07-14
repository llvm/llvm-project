#!/bin/sh

# Usage:
# utils/change-addr-space.sh src  : apply utils/add_amdgiz.sed
#                                adopt generic address space is address space 0
# utils/change-addr-space.sh src x : apply utils/remove_amdgiz.sed
#                                adopt generic address space is address space 4

tmpfile=/tmp/cas$$.sed
if [ $# -lt 3 ]; then
  echo "/target triple/s/\\\"amdgcn--amdhsa\\\"/\\\"${1}\\\"/" >$tmpfile
  cat $2/add_amdgiz.sed >>$tmpfile
else
  echo "/target triple/s/\\\"${1}\\\"/\\\"amdgcn--amdhsa\\\"/" >$tmpfile
  cat $2/remove_amdgiz.sed >>$tmpfile
fi

find . -name "*.ll" | xargs sed -i -f "$tmpfile"
rm $tmpfile
