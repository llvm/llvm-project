// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: target={{.*}}-aix{{.*}}

// Check that we discover dependency on a precompiled module when it's imported
// by a **submodule** instead of a top-level module.

// RUN: rm -rf %t && mkdir %t
// RUN: cp %S/Inputs/modules-pch-common-via-submodule/* %t

// Scan dependencies of the PCH:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch-common-via-submodule/cdb_pch.json > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -module-files-dir %t/build > %t/result_pch.json
// RUN: cat %t/result_pch.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -check-prefix=CHECK-PCH
//
// CHECK-PCH:      {
// CHECK-PCH-NEXT:   "modules": [
// CHECK-PCH-NEXT:     {
// CHECK-PCH-NEXT:       "clang-module-deps": [],
// CHECK-PCH-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH:            ],
// CHECK-PCH-NEXT:       "context-hash": "[[HASH_MOD_COMMON:.*]]",
// CHECK-PCH-NEXT:       "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-PCH-NEXT:         "[[PREFIX]]/mod_common.h"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "link-libraries": [],
// CHECK-PCH-NEXT:       "name": "ModCommon"
// CHECK-PCH-NEXT:     }
// CHECK-PCH-NEXT:   ],
// CHECK-PCH-NEXT:   "translation-units": [
// CHECK-PCH-NEXT:     {
// CHECK-PCH:            "clang-context-hash": "[[HASH_PCH:.*]]",
// CHECK-PCH-NEXT:       "clang-module-deps": [
// CHECK-PCH-NEXT:         {
// CHECK-PCH-NEXT:           "context-hash": "[[HASH_MOD_COMMON]]",
// CHECK-PCH-NEXT:           "module-name": "ModCommon"
// CHECK-PCH-NEXT:         }
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "command-line": [
// CHECK-PCH:            ],
// CHECK-PCH:            "file-deps": [
// CHECK-PCH-NEXT:         "[[PREFIX]]/pch.h"
// CHECK-PCH-NEXT:       ],
// CHECK-PCH-NEXT:       "input-file": "[[PREFIX]]/pch.h"
// CHECK-PCH-NEXT:     }

// Explicitly build the PCH:
//
// RUN: %deps-to-rsp %t/result_pch.json --module-name=ModCommon > %t/mod_common.cc1.rsp
// RUN: %deps-to-rsp %t/result_pch.json --tu-index=0 > %t/pch.rsp
//
// RUN: %clang @%t/mod_common.cc1.rsp
// RUN: %clang @%t/pch.rsp

// Scan dependencies of the TU:
//
// RUN: sed "s|DIR|%/t|g" %S/Inputs/modules-pch-common-via-submodule/cdb_tu.json > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -module-files-dir %t/build > %t/result_tu.json
// RUN: cat %t/result_tu.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t -check-prefix=CHECK-TU
//
// CHECK-TU:      {
// CHECK-TU-NEXT:   "modules": [
// CHECK-TU-NEXT:     {
// CHECK-TU-NEXT:       "clang-module-deps": [],
// CHECK-TU-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU:            ],
// CHECK-TU-NEXT:       "context-hash": "[[HASH_MOD_TU:.*]]",
// CHECK-TU-NEXT:       "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-TU-NEXT:         "[[PREFIX]]/mod_tu.h",
// CHECK-TU-NEXT:         "[[PREFIX]]/mod_tu_sub.h"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "link-libraries": [],
// CHECK-TU-NEXT:       "name": "ModTU"
// CHECK-TU-NEXT:     }
// CHECK-TU-NEXT:   ],
// CHECK-TU-NEXT:   "translation-units": [
// CHECK-TU-NEXT:     {
// CHECK-TU:            "clang-context-hash": "[[HASH_TU:.*]]",
// CHECK-TU-NEXT:       "clang-module-deps": [
// CHECK-TU-NEXT:         {
// CHECK-TU-NEXT:           "context-hash": "[[HASH_MOD_TU]]"
// CHECK-TU-NEXT:           "module-name": "ModTU"
// CHECK-TU-NEXT:         }
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "command-line": [
// CHECK-TU:            ],
// CHECK-TU:            "file-deps": [
// CHECK-TU-NEXT:         "[[PREFIX]]/tu.c",
// CHECK-TU-NEXT:         "[[PREFIX]]/pch.h.pch"
// CHECK-TU-NEXT:       ],
// CHECK-TU-NEXT:       "input-file": "[[PREFIX]]/tu.c"
// CHECK-TU-NEXT:     }

// Explicitly build the TU:
//
// RUN: %deps-to-rsp %t/result_tu.json --module-name=ModTU > %t/mod_tu.cc1.rsp
// RUN: %deps-to-rsp %t/result_tu.json --tu-index=0 > %t/tu.rsp
//
// RUN: %clang @%t/mod_tu.cc1.rsp
// RUN: %clang @%t/tu.rsp
