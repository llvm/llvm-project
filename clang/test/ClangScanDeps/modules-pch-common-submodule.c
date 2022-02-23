// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: aix

// Check that when depending on a precompiled module, we depend on the
// **top-level** module. Submodules don't have some information present (for
// example the path to the modulemap file) and depending on them might cause
// problems in the dependency scanner (e.g. generating empty `-fmodule-map-file=`
// arguments).

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/modules-pch-common-submodule/* %t

// Scan dependencies of the PCH:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch-common-submodule/cdb_pch.json > %t/cdb.json
// RUN: echo -%t > %t/result_pch.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build >> %t/result_pch.json
// RUN: cat %t/result_pch.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK-PCH
//
// CHECK-PCH:      -[[PREFIX:.*]]
// CHECK-PCH-NEXT: {
// CHECK-PCH-NEXT:   "modules": [
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-module-deps": [],
// CHECK-PCH-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH-NEXT:         "-cc1"
// CHECK-PCH:              "-emit-module"
// CHECK-PCH:              "-fmodules"
// CHECK-PCH:              "-fmodule-name=ModCommon"
// CHECK-PCH:              "-fno-implicit-modules"
// CHECK-PCH:            ],
// CHECK-PCH-NEXT:       "context-hash": "[[HASH_MOD_COMMON:.*]]",
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/mod_common.h",
// CHECK-PCH-NEXT:         "[[PREFIX]]/mod_common_sub.h",
// CHECK-PCH-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "name": "ModCommon"
// CHECK-PCH-NEXT:     }
// CHECK-PCH-NEXT:   ],
// CHECK-PCH-NEXT:   "translation-units": [
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-context-hash": "[[HASH_PCH:.*]]",
// CHECK-PCH-NEXT:       "clang-module-deps": [
// CHECK-PCH-NEXT:         {
// CHECK-PCH-NEXT:           "context-hash": "[[HASH_MOD_COMMON]]",
// CHECK-PCH-NEXT:           "module-name": "ModCommon"
// CHECK-PCH-NEXT:         }
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH:              "-fno-implicit-modules"
// CHECK-PCH-NEXT:         "-fno-implicit-module-maps"
// CHECK-PCH-NEXT:         "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_COMMON]]/ModCommon-{{.*}}.pcm"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/pch.h"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "input-file": "[[PREFIX]]/pch.h"
// CHECK-PCH-NEXT:     }
// CHECK-PCH-NEXT:   ]
// CHECK-PCH-NEXT: }

// Explicitly build the PCH:
//
// RUN: tail -n +2 %t/result_pch.json > %t/result_pch_stripped.json
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_pch_stripped.json \
// RUN:   --module-name=ModCommon > %t/mod_common.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_pch_stripped.json \
// RUN:   --tu-index=0 > %t/pch.rsp
//
// RUN: %clang @%t/mod_common.cc1.rsp
// RUN: %clang @%t/pch.rsp

// Scan dependencies of the TU:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch-common-submodule/cdb_tu.json > %t/cdb.json
// RUN: echo -%t > %t/result_tu.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -generate-modules-path-args -module-files-dir %t/build >> %t/result_tu.json
// RUN: cat %t/result_tu.json | sed 's:\\\\\?:/:g' | FileCheck %s -check-prefix=CHECK-TU
//
// CHECK-TU:      -[[PREFIX:.*]]
// CHECK-TU-NEXT: {
// CHECK-TU-NEXT:   "modules": [
// CHECK-TU-NEXT:     {
// CHECK-TU-NEXT:       "clang-module-deps": [],
// CHECK-TU-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU-NEXT:         "-cc1"
// CHECK-TU:              "-emit-module"
// CHECK-TU:              "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_COMMON:.*]]/ModCommon-{{.*}}.pcm"
// CHECK-TU:              "-fmodules"
// CHECK-TU:              "-fmodule-name=ModTU"
// CHECK-TU:              "-fno-implicit-modules"
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
// CHECK-TU-NEXT:           "context-hash": "[[HASH_MOD_TU]]"
// CHECK-TU-NEXT:           "module-name": "ModTU"
// CHECK-TU-NEXT:         }
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU:              "-fno-implicit-modules",
// CHECK-TU-NEXT:         "-fno-implicit-module-maps",
// CHECK-TU-NEXT:         "-fmodule-file=[[PREFIX]]/build/[[HASH_MOD_TU:.*]]/ModTU-{{.*}}.pcm"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/tu.c",
// CHECK-TU-NEXT:         "[[PREFIX]]/pch.h.gch"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-TU-NEXT:     }
// CHECK-TU-NEXT:   ]
// CHECK-TU-NEXT: }

// Explicitly build the TU:
//
// RUN: tail -n +2 %t/result_tu.json > %t/result_tu_stripped.json
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_tu_stripped.json \
// RUN:   --module-name=ModTU > %t/mod_tu.cc1.rsp
// RUN: %python %S/../../utils/module-deps-to-rsp.py %t/result_tu_stripped.json \
// RUN:   --tu-index=0 > %t/tu.rsp
//
// RUN: %clang @%t/mod_tu.cc1.rsp
// RUN: %clang @%t/tu.rsp
