// If secondary output files such as .d are enabled, ensure it affects the
// module context hash since it may impact the resulting module build commands.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 -generate-modules-path-args \
// RUN:   -format experimental-full > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK:          {
// CHECK:            "command-line": [
// CHECK:              "-dependency-file"
// CHECK:            ]
// CHECK:            "context-hash": "[[HASH1:.*]]"
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK-NOT:          "-dependency-file"
// CHECK:            ]
// CHECK:            "context-hash": "[[HASH2:.*]]"
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
// CHECK-NEXT:       "command-line": [
// CHECK:              "-MF"
// CHECK:            ]
// CHECK:            "input-file": "{{.*}}tu1.c"
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH2]]"
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-MF"
// CHECK:            ]
// CHECK:            "input-file": "{{.*}}tu2.c"

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -MD -MF DIR/tu1.d -fsyntax-only DIR/tu1.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu1.c"
  },
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu2.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu2.c"
  },
]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu1.c
#include "Mod.h"

//--- tu2.c
#include "Mod.h"
