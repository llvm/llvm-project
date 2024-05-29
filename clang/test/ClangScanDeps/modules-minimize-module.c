// RUN: rm -rf %t
// RUN: split-file %s %t

// This test checks that source files of modules undergo dependency directives
// scan. If a.h would not, the scan would fail when lexing `#error`.

//--- module.modulemap
module A { header "a.h" }

//--- a.h
#error blah

//--- tu.c
#include "a.h"

//--- cdb.json.in
[{
  "directory": "DIR",
  "file": "DIR/tu.c",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/deps.json
