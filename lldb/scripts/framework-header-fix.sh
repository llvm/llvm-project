#!/bin/sh
# Usage: framework-header-fix.sh <source header dir> <LLDB Version>

set -e

for file in `find $1 -name "*.h"`
do
  /usr/bin/sed -i.bak 's/\(#include\)[ ]*"lldb\/\(API\/\)\{0,1\}\(.*\)"/\1 <LLDB\/\3>/1' "$file"
  /usr/bin/sed -i.bak 's|<LLDB/Utility|<LLDB|' "$file"
  rm -f "$file.bak"
done
