// RUN: rm -rf %t && mkdir %t
// RUN: cp -r %S/Inputs/modules-context-hash/* %t

// Check that the scanner reports the same module as distinct dependencies when
// a single translation unit gets compiled with multiple command-lines that
// produce different **strict** context hashes.

// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-context-hash/cdb_a.json.template > %t/cdb_a.json
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-context-hash/cdb_b.json.template > %t/cdb_b.json

// We run two separate scans. The context hash for "a" and "b" can differ between
// systems. If we'd scan both Clang invocations in a single run, the order of JSON
// entities would be non-deterministic. To prevent this, run the scans separately
// and verify that the context hashes differ with a single FileCheck invocation.
//
// RUN: echo -%t > %t/result.json
// RUN: clang-scan-deps -compilation-database %t/cdb_a.json -format experimental-full -j 1 >> %t/result.json
// RUN: clang-scan-deps -compilation-database %t/cdb_b.json -format experimental-full -j 1 >> %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK

// CHECK:      -[[PREFIX:.*]]
// CHECK-NEXT: {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-emit-module"
// CHECK:              "-I"
// CHECK:              "[[PREFIX]]/a"
// CHECK:              "-fmodule-name=mod"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MOD_A:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/a/dep.h",
// CHECK-NEXT:         "[[PREFIX]]/mod.h",
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD_A]]",
// CHECK-NEXT:           "module-name": "mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1"
// CHECK:              "-emit-module"
// CHECK:              "-I"
// CHECK:              "[[PREFIX]]/b"
// CHECK:              "-fmodule-name=mod"
// CHECK:            ],
// CHECK-NOT:        "context-hash": "[[HASH_MOD_A]]",
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/b/dep.h",
// CHECK-NEXT:         "[[PREFIX]]/mod.h",
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NOT:            "context-hash": "[[HASH_MOD_A]]",
// CHECK:                "module-name": "mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-fno-implicit-modules",
// CHECK-NEXT:         "-fno-implicit-module-maps"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
