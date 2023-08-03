// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%t|g" -e "s|CLANG|%clang|g" -e "s|SDK|%S/Inputs/SDK|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -cas-path %t/cas \
// RUN:   -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc > %t/result.txt
// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK

// CHECK:      {{.*}} - [[PREFIX]]/t.c
// CHECK-NOT: [[PREFIX]]
// CHECK-NOT: [[SDK_PREFIX]]
// CHECK: /^src{{[/\\]}}t.c
// CHECK: /^src{{[/\\]}}top.h
// CHECK: /^tc{{[/\\]}}lib{{[/\\]}}clang{{[/\\]}}{{.*}}{{[/\\]}}include{{[/\\]}}stdarg.h
// CHECK: /^sdk{{[/\\]}}usr{{[/\\]}}include{{[/\\]}}stdlib.h

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc > %t/deps.json

// RUN: cat %t/result.txt > %t/full.txt
// RUN: echo "FULL DEPS START" >> %t/full.txt
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' >> %t/full.txt

// RUN: FileCheck %s -DPREFIX=%/t -DSDK_PREFIX=%S/Inputs/SDK -check-prefix=FULL -input-file %t/full.txt

// Capture the tree id from experimental-include-tree ; ensure that it matches
// the result from experimental-full.
// FULL: [[TREE_ID:llvmcas://[[:xdigit:]]+]] - [[PREFIX]]/t.c
// FULL: FULL DEPS START

// FULL-NEXT: {
// FULL-NEXT:   "modules": []
// FULL-NEXT:   "translation-units": [
// FULL-NEXT:     {
// FULL-NEXT:       "commands": [
// FULL-NEXT:         {
// FULL:                "clang-module-deps": []
// FULL:                "command-line": [
// FULL-NEXT:             "-cc1"
// FULL:                  "-fcas-path"
// FULL-NEXT:             "[[PREFIX]]/cas"
// FULL:                  "-disable-free"
// FULL:                  "-fcas-include-tree"
// FULL-NEXT:             "[[TREE_ID]]"
// FULL:                  "-fcache-compile-job"
// FULL:                  "-fsyntax-only"
// FULL:                  "-x"
// FULL-NEXT:             "c"
// FULL:                  "-isysroot"
// FULL-NEXT:             "/^sdk"
// FULL:                ]
// FULL:                "file-deps": [
// FULL-DAG:              "[[PREFIX]]/t.c"
// FULL-DAG:              "[[PREFIX]]/top.h"
// FULL-DAG:              "{{.*}}/stdarg.h"
// FULL-DAG:              "[[SDK_PREFIX]]/usr/include/stdlib.h"
// FULL:                ]
// FULL:                "input-file": "[[PREFIX]]/t.c"
// FULL:              }
// FULL:            ]
// FULL:          }
// FULL:        ]
// FULL:      }

// Build the include-tree command
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "CLANG -fsyntax-only DIR/t.c -target x86_64-apple-macos11 -isysroot SDK",
    "file": "DIR/t.c"
  }
]

//--- t.c
#include "top.h"

//--- top.h
#include <stdarg.h>
#include <stdlib.h>
