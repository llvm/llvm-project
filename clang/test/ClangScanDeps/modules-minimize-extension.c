// RUN: rm -rf %t
// RUN: split-file %s %t

// This test checks that source files with uncommon extensions still undergo
// dependency directives scan. If header.pch would not and b.h would, the scan
// would fail when parsing `void function(B)` and not knowing the symbol B.

//--- module.modulemap
module __PCH { header "header.pch" }
module B { header "b.h" }

//--- header.pch
#include "b.h"
void function(B);

//--- b.h
typedef int B;

//--- tu.c
int main() {
  function(0);
  return 0;
}

//--- cdb.json.in
[{
  "directory": "DIR",
  "file": "DIR/tu.c",
  "command": "clang -c DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -fimplicit-module-maps -include DIR/header.pch"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/deps.json
