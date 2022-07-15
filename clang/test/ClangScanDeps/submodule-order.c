// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -generate-modules-path-args > %t/deps1.json
// RUN: mv %t/tu2.c %t/tu.c
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -generate-modules-path-args > %t/deps2.json
// RUN: diff -u %t/deps1.json %t/deps2.json
// RUN: FileCheck %s < %t/deps1.json

// CHECK: "-fmodule-file={{.*}}Indirect1
// CHECK-NOT: "-fmodule-file={{.*}}Indirect
// CHECK: "-fmodule-file={{.*}}Indirect2
// CHECK-NOT: "-fmodule-file={{.*}}Indirect

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
  "file": "DIR/tu.c"
}]

//--- module.modulemap
module Indirect1 { header "Indirect1.h" }
module Indirect2 { header "Indirect2.h" }
module Mod {
  umbrella "Mod"
  module * { export * }
}

//--- Indirect1.h
void indirect1(void);

//--- Indirect2.h
void indirect2(void);

//--- Mod/SubMod1.h
#include "../Indirect1.h"

//--- Mod/SubMod2.h
#include "../Indirect2.h"

//--- tu.c
#include "Mod/SubMod1.h"
#include "Mod/SubMod2.h"
void tu1(void) {
  indirect1();
  indirect2();
}

//--- tu2.c
#include "Mod/SubMod2.h"
#include "Mod/SubMod1.h"
void tu1(void) {
  indirect1();
  indirect2();
}