// Some command-line arguments used for compiling translation units are not
// compatible with the semantics of modules or are likely to differ between
// identical modules discovered from different translation units. This test
// checks such arguments are removed from the command-lines: '-include',
// '-dwarf-debug-flag' and '-main-file-name'.

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/removed-args/* %t

// RUN: sed "s|DIR|%/t|g" %S/Inputs/removed-args/cdb.json.template > %t/cdb.json
// RUN: echo -%t > %t/result.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full >> %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s
//
// CHECK:      -[[PREFIX:.*]]
// CHECK-NEXT: {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK-NOT:          "-dwarf-debug-flags"
// CHECK-NOT:          "-main-file-name"
// CHECK-NOT:          "-include"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MOD_HEADER:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/mod_header.h",
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "ModHeader"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK-NOT:          "-include"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MOD_TU:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/mod_tu.h",
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "ModTU"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "[[HASH_TU:.*]]",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD_HEADER]]",
// CHECK-NEXT:           "module-name": "ModHeader"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD_TU]]",
// CHECK-NEXT:           "module-name": "ModTU"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK:          }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
