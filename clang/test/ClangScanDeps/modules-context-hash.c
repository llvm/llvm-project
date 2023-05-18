// RUN: rm -rf %t && mkdir %t
// RUN: cp -r %S/Inputs/modules-context-hash/* %t

// Check that the scanner reports the same module as distinct dependencies when
// a single translation unit gets compiled with multiple command-lines that
// produce different **strict** context hashes.

// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-context-hash/cdb_a.json.template > %t/cdb_a.json
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-context-hash/cdb_b.json.template > %t/cdb_b.json
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-context-hash/cdb_b2.json.template > %t/cdb_b2.json

// We run separate scans. The context hash for "a" and "b" can differ between
// systems. If we'd scan both Clang invocations in a single run, the order of JSON
// entities would be non-deterministic. To prevent this, run the scans separately
// and verify that the context hashes differ with a single FileCheck invocation.
//
// RUN: clang-scan-deps -compilation-database %t/cdb_a.json -format experimental-full -j 1 >  %t/result_a.json
// RUN: clang-scan-deps -compilation-database %t/cdb_b.json -format experimental-full -j 1 > %t/result_b.json
// RUN: clang-scan-deps -compilation-database %t/cdb_b2.json -format experimental-full -j 1 > %t/result_b2.json
// RUN: cat %t/result_a.json %t/result_b.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -check-prefix=CHECK
// RUN: cat %t/result_b.json %t/result_b2.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -check-prefix=FLAG_ONLY

// CHECK:      {
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
// CHECK:            "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MOD_A]]",
// CHECK-NEXT:           "module-name": "mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }

// CHECK:       "modules": [
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
// CHECK:            "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NOT:            "context-hash": "[[HASH_MOD_A]]",
// CHECK:                "module-name": "mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:     }

// B and B2 only differ by -fapplication-extension

// FLAG_ONLY:       "modules": [
// FLAG_ONLY-NEXT:     {
// FLAG_ONLY:            "context-hash": "[[HASH_MOD_B1:.*]]"
// FLAG_ONLY-NOT:        "-fapplication-extension"

// FLAG_ONLY:       "modules": [
// FLAG_ONLY-NEXT:     {
// FLAG_ONLY-NOT:        "context-hash": "[[HASH_MOD_B1]]"
// FLAG_ONLY:            "-fapplication-extension"
// FLAG_ONLY:       "translation-units": [
// FLAG_ONLY-NOT:        "context-hash": "[[HASH_MOD_B1]]"
