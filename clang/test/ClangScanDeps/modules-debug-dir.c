// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full > %t/result.json

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -g -fdebug-info-for-profiling DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -IDIR/include/ -o DIR/tu.o",
  "file": "DIR/tu.c"
}]

//--- include/module.modulemap
module mod {
  header "mod.h"
}

//--- include/mod.h

//--- tu.c
#include "mod.h"
