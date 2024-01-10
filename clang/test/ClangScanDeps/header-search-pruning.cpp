// RUN: rm -rf %t && mkdir -p %t
// RUN: cp -r %S/Inputs/header-search-pruning/* %t
// RUN: cp %S/header-search-pruning.cpp %t/header-search-pruning.cpp
// RUN: sed -e "s|DIR|%/t|g" -e "s|DEFINES|-DINCLUDE_A|g"             %S/Inputs/header-search-pruning/cdb.json > %t/cdb_a.json
// RUN: sed -e "s|DIR|%/t|g" -e "s|DEFINES|-DINCLUDE_B|g"             %S/Inputs/header-search-pruning/cdb.json > %t/cdb_b.json
// RUN: sed -e "s|DIR|%/t|g" -e "s|DEFINES|-DINCLUDE_A -DINCLUDE_B|g" %S/Inputs/header-search-pruning/cdb.json > %t/cdb_ab.json
//
// RUN: clang-scan-deps -compilation-database %t/cdb_a.json -format experimental-full -optimize-args=header-search >> %t/result_a.json
// RUN: cat %t/result_a.json | sed 's:\\\\\?:/:g' | FileCheck --check-prefixes=CHECK_A %s
//
// RUN: clang-scan-deps -compilation-database %t/cdb_b.json -format experimental-full -optimize-args=header-search >> %t/result_b.json
// RUN: cat %t/result_b.json | sed 's:\\\\\?:/:g' | FileCheck --check-prefixes=CHECK_B %s
//
// RUN: clang-scan-deps -compilation-database %t/cdb_ab.json -format experimental-full -optimize-args=header-search >> %t/result_ab.json
// RUN: cat %t/result_ab.json | sed 's:\\\\\?:/:g' | FileCheck --check-prefixes=CHECK_AB %s

#include "mod.h"

// CHECK_A:        {
// CHECK_A-NEXT:     "modules": [
// CHECK_A-NEXT:       {
// CHECK_A-NEXT:         "clang-module-deps": [],
// CHECK_A-NEXT:         "clang-modulemap-file": "{{.*}}",
// CHECK_A-NEXT:         "command-line": [
// CHECK_A-NEXT:           "-cc1"
// CHECK_A:                "-I",
// CHECK_A-NEXT:           "begin",
// CHECK_A-NEXT:           "-I",
// CHECK_A-NEXT:           "a",
// CHECK_A-NEXT:           "-I",
// CHECK_A-NEXT:           "end"
// CHECK_A:              ],
// CHECK_A-NEXT:         "context-hash": "{{.*}}",
// CHECK_A-NEXT:         "file-deps": [
// CHECK_A:              ],
// CHECK_A-NEXT:         "name": "mod"
// CHECK_A-NEXT:       }
// CHECK_A-NEXT:     ]
// CHECK_A:        }

// CHECK_B:        {
// CHECK_B-NEXT:     "modules": [
// CHECK_B-NEXT:       {
// CHECK_B-NEXT:         "clang-module-deps": [],
// CHECK_B-NEXT:         "clang-modulemap-file": "{{.*}}",
// CHECK_B-NEXT:         "command-line": [
// CHECK_B-NEXT:           "-cc1"
// CHECK_B:                "-I",
// CHECK_B-NEXT:           "begin",
// CHECK_B-NEXT:           "-I",
// CHECK_B-NEXT:           "b",
// CHECK_B-NEXT:           "-I",
// CHECK_B-NEXT:           "end"
// CHECK_B:              ],
// CHECK_B-NEXT:         "context-hash": "{{.*}}",
// CHECK_B-NEXT:         "file-deps": [
// CHECK_B:              ],
// CHECK_B-NEXT:         "name": "mod"
// CHECK_B-NEXT:       }
// CHECK_B-NEXT:     ]
// CHECK_B:        }

// CHECK_AB:       {
// CHECK_AB-NEXT:    "modules": [
// CHECK_AB-NEXT:      {
// CHECK_AB-NEXT:        "clang-module-deps": [],
// CHECK_AB-NEXT:        "clang-modulemap-file": "{{.*}}",
// CHECK_AB-NEXT:        "command-line": [
// CHECK_AB-NEXT:          "-cc1"
// CHECK_AB:               "-I",
// CHECK_AB-NEXT:          "begin",
// CHECK_AB-NEXT:          "-I",
// CHECK_AB-NEXT:          "a",
// CHECK_AB-NEXT:          "-I",
// CHECK_AB-NEXT:          "b",
// CHECK_AB-NEXT:          "-I",
// CHECK_AB-NEXT:          "end"
// CHECK_AB:             ],
// CHECK_AB-NEXT:        "context-hash": "{{.*}}",
// CHECK_AB-NEXT:        "file-deps": [
// CHECK_AB:             ],
// CHECK_AB-NEXT:        "name": "mod"
// CHECK_AB-NEXT:      }
// CHECK_AB-NEXT:    ]
// CHECK_AB:       }
