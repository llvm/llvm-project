// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/diagnostics/* %t

// RUN: sed "s|DIR|%/t|g" %S/Inputs/diagnostics/cdb.json.template > %t/cdb.json
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
// CHECK-NEXT:       "name": "mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:.*]],
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD]]",
// CHECK-NEXT:           "module-name": "mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fno-implicit-modules"
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
