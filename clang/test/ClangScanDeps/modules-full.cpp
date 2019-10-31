// RUN: rm -rf %t.dir
// RUN: rm -rf %t.cdb
// RUN: rm -rf %t.module-cache
// RUN: mkdir -p %t.dir
// RUN: cp %s %t.dir/modules_cdb_input.cpp
// RUN: cp %s %t.dir/modules_cdb_input2.cpp
// RUN: mkdir %t.dir/Inputs
// RUN: cp %S/Inputs/header.h %t.dir/Inputs/header.h
// RUN: cp %S/Inputs/header2.h %t.dir/Inputs/header2.h
// RUN: cp %S/Inputs/module.modulemap %t.dir/Inputs/module.modulemap
// RUN: sed -e "s|DIR|%/t.dir|g" %S/Inputs/modules_cdb.json > %t.cdb
//
// RUN: echo %t.dir > %t.result
// RUN: clang-scan-deps -compilation-database %t.cdb -j 1 \
// RUN:   -mode preprocess-minimized-sources -format experimental-full >> %t.result
// RUN: cat %t.result | FileCheck --check-prefixes=CHECK %s

// FIXME: Backslash issues.
// XFAIL: system-windows

#include "header.h"

// CHECK: [[PREFIX:(.*[/\\])+[a-zA-Z0-9.-]+]]
// CHECK-NEXT:     {
// CHECK-NEXT:  "clang-context-hash": "[[CONTEXT_HASH:[A-Z0-9]+]]",
// CHECK-NEXT:  "clang-module-deps": [
// CHECK-NEXT:    "header1"
// CHECK-NEXT:  ],
// CHECK-NEXT:  "clang-modules": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "clang-module-deps": [
// CHECK-NEXT:        "header2"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "clang-modulemap-file": "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module.modulemap",
// CHECK-NEXT:      "file-deps": [
// CHECK-NEXT:        "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}header.h",
// CHECK-NEXT:        "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module.modulemap"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "name": "header1"
// CHECK-NEXT:    },
// CHECK-NEXT:    {
// CHECK-NEXT:      "clang-module-deps": [],
// CHECK-NEXT:      "clang-modulemap-file": "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module.modulemap",
// CHECK-NEXT:      "file-deps": [
// CHECK-NEXT:        "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}header2.h",
// CHECK-NEXT:        "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module.modulemap"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "name": "header2"
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "file-deps": [
// CHECK-NEXT:    "[[PREFIX]]{{[/\\]}}modules_cdb_input2.cpp"
// CHECK-NEXT:  ],
// CHECK-NEXT:  "input-file": "[[PREFIX]]{{[/\\]}}modules_cdb_input2.cpp"
// CHECK-NEXT:},
// CHECK-NEXT:{
// CHECK-NOT:   "clang-context-hash": "[[CONTEXT_HASH]]",
// CHECK-NEXT:  "clang-context-hash": "{{[A-Z0-9]+}}",
// CHECK-NEXT:  "clang-module-deps": [
// CHECK-NEXT:    "header1"
// CHECK-NEXT:  ],
// CHECK-NEXT:  "clang-modules": [
// CHECK-NEXT:    {
// CHECK-NEXT:      "clang-module-deps": [],
// CHECK-NEXT:      "clang-modulemap-file": "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module.modulemap",
// CHECK-NEXT:      "file-deps": [
// CHECK-NEXT:        "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}header.h",
// CHECK-NEXT:        "[[PREFIX]]{{[/\\]}}Inputs{{[/\\]}}module.modulemap"
// CHECK-NEXT:      ],
// CHECK-NEXT:      "name": "header1"
// CHECK-NEXT:    }
// CHECK-NEXT:  ],
// CHECK-NEXT:  "file-deps": [
// CHECK-NEXT:    "[[PREFIX]]{{[/\\]}}modules_cdb_input.cpp"
// CHECK-NEXT:  ],
// CHECK-NEXT:  "input-file": "[[PREFIX]]{{[/\\]}}modules_cdb_input.cpp"
// CHECK-NEXT:},
