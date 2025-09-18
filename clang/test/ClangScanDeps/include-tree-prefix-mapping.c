// REQUIRES: ondisk_cas
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" -e "s|CLANG|%/ncclang|g" -e "s|SDK|%/S/Inputs/SDK|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -cas-path %t/cas \
// RUN:   -prefix-map=%t=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc > %t/result.txt
// RUN: cat %t/result.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize SDK_PREFIX=%/S/Inputs/SDK --sanitize ROOT^=%/root^ %s

// CHECK:      {{.*}} - PREFIX{{/|\\}}t.c
// CHECK-NOT: PREFIX
// CHECK-NOT: SDK_PREFIX
// CHECK: ROOT^src{{[/\\]}}t.c
// CHECK: ROOT^src{{[/\\]}}top.h
// CHECK: ROOT^tc{{[/\\]}}lib{{[/\\]}}clang{{[/\\]}}{{.*}}{{[/\\]}}include{{[/\\]}}stdarg.h
// CHECK: ROOT^sdk{{[/\\]}}usr{{[/\\]}}include{{[/\\]}}stdlib.h

// RUN: clang-scan-deps -compilation-database %t/cdb.json \
// RUN:   -format experimental-include-tree-full -cas-path %t/cas \
// RUN:   -prefix-map=%t=%/root^src -prefix-map-sdk=%/root^sdk -prefix-map-toolchain=%/root^tc > %t/deps.json

// RUN: cat %t/result.txt > %t/full.txt
// RUN: echo "FULL DEPS START" >> %t/full.txt
// RUN: cat %t/deps.json >> %t/full.txt

// RUN: cat %t/full.txt | %PathSanitizingFileCheck --sanitize PREFIX=%/t --sanitize SDK_PREFIX=%/S/Inputs/SDK --sanitize ROOT^=%/root^ --enable-yaml-compatibility %s -check-prefix=FULL

// Capture the tree id from experimental-include-tree ; ensure that it matches
// the result from experimental-full.
// FULL: [[TREE_ID:llvmcas://[[:xdigit:]]+]] - PREFIX{{/|\\}}t.c
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
// FULL-NEXT:             "PREFIX{{/|\\\\}}cas"
// FULL:                  "-disable-free"
// FULL:                  "-fcas-include-tree"
// FULL-NEXT:             "[[TREE_ID]]"
// FULL:                  "-fcache-compile-job"
// FULL:                  "-fsyntax-only"
// FULL:                  "-x"
// FULL-NEXT:             "c"
// FULL:                  "-isysroot"
// FULL-NEXT:             "ROOT^sdk"
// FULL:                ]
// FULL:                "file-deps": [
// FULL-DAG:              "PREFIX{{/|\\\\}}t.c"
// FULL-DAG:              "PREFIX{{/|\\\\}}top.h"
// FULL-DAG:              "{{.*}}{{/|\\\\}}stdarg.h"
// FULL-DAG:              "SDK_PREFIX{{/|\\\\}}usr{{/|\\\\}}include{{/|\\\\}}stdlib.h"
// FULL:                ]
// FULL:                "input-file": "PREFIX{{/|\\\\}}t.c"
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
