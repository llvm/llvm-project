// Ensure the path to the modulemap input is included in the module context hash
// irrespective of other TU command-line arguments, as it effects the canonical
// module build command. In this test we use the difference in spelling between
// module.modulemap and module.map, but it also applies to situations such as
// differences in case-insensitive paths if they are not canonicalized away.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full > %t/deps.json

// RUN: mv %t/module.modulemap %t/module.map
// RUN: echo 'AFTER_MOVE' >> %t/deps.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full >> %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK:          {
// CHECK:            "command-line": [
// CHECK:              "{{.*}}module.modulemap"
// CHECK:            ]
// CHECK:            "context-hash": "[[HASH1:.*]]"
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH1]]"
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-LABEL: AFTER_MOVE
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK:          {
// CHECK-NOT: [[HASH1]]
// CHECK:            "command-line": [
// CHECK:              "{{.*}}module.map"
// CHECK:            ]
// CHECK-NOT: [[HASH1]]
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash":
// CHECK-NOT: [[HASH1]]
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu.c"
  }
]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu.c
#include "Mod.h"
