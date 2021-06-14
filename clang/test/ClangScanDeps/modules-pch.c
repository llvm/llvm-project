// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/modules-pch/* %t

// Explicitly build the PCH:
//
// RUN: %clang -x c-header %t/pch.h -fmodules -gmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -o %t/pch.h.gch

// Scan dependencies of the TU:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch/cdb_tu.json > %t/cdb.json
// RUN: echo -%t > %t/result_tu.json
// FIXME: Make this work with '-mode preprocess-minimized-sources'.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build -mode preprocess >> %t/result_tu.json
// RUN: cat %t/result_tu.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK-TU
//
// CHECK-TU:      -[[PREFIX:.*]]
// CHECK-TU-NEXT: {
// CHECK-TU-NEXT:   "modules": [
// CHECK-TU-NEXT:     {
// CHECK-TU-NEXT:       "clang-module-deps": [],
// CHECK-TU-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU-NEXT:         "-cc1",
// CHECK-TU:              "-emit-module",
// CHECK-TU:              "-fmodule-name=ModTU",
// CHECK-TU:              "-fno-implicit-modules",
// CHECK-TU:            ],
// CHECK-TU-NEXT:       "context-hash": "[[HASH_MOD_TU:.*]]",
// CHECK-TU-NEXT:       "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/mod_tu.h",
// CHECK-TU-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "name": "ModTU"
// CHECK-TU-NEXT:     }
// CHECK-TU-NEXT:   ],
// CHECK-TU-NEXT:   "translation-units": [
// CHECK-TU-NEXT:     {
// CHECK-TU-NEXT:       "clang-context-hash": "[[HASH_TU:.*]]",
// CHECK-TU-NEXT:       "clang-module-deps": [
// CHECK-TU-NEXT:         {
// CHECK-TU-NEXT:           "context-hash": "[[HASH_MOD_TU]]",
// CHECK-TU-NEXT:           "module-name": "ModTU"
// CHECK-TU-NEXT:         }
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU-NEXT:         "-fno-implicit-modules",
// CHECK-TU-NEXT:         "-fno-implicit-module-maps",
// CHECK-TU-NEXT:         "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_TU]]/ModTU-{{.*}}.pcm",
// CHECK-TU-NEXT:         "-fmodule-map-file=[[PREFIX]]/module.modulemap"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/tu.c",
// CHECK-TU-NEXT:         "[[PREFIX]]/pch.h.gch"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-TU-NEXT:     }
// CHECK-TU-NEXT:   ]
// CHECK-TU-NEXT: }
