// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -fmodules -target i386-apple-ios14.0-simulator -fmodules-cache-path=DIR/cache -Wno-error=invalid-ios-deployment-target -o DIR/tu.o",
    "file": "DIR/tu.c"
  }
]
//--- mod.h
//--- module.modulemap
module mod { header "mod.h" }
//--- tu.c
#include "mod.h"

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full 2>&1 > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Check that the '-Wno-error=invalid-ios-deployment-target' option is being
// respected and invalid arguments like '-target i386-apple-ios14.0-simulator'
// do not result in an error.

// CHECK-NOT:  error:
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MOD:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/mod.h"
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "[[HASH_TU:.*]],
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD]]",
// CHECK-NEXT:           "module-name": "mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fimplicit-modules"
// CHECK-NOT:          "-fimplicit-module-maps"
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }
