// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%t|g" -e "s|CLANG|%clang|g" -e "s|SDK|%S/Inputs/SDK|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -in-memory-cas \
// RUN:   -prefix-map=%t=/^src -prefix-map-sdk=/^sdk -prefix-map-toolchain=/^tc > %t/result.txt
// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%t -DSDK_PREFIX=%S/Inputs/SDK

// CHECK:      {{.*}} - [[PREFIX]]/t.c
// CHECK-NOT: [[PREFIX]]
// CHECK-NOT: [[SDK_PREFIX]]
// CHECK: /^src{{[/\\]}}t.c
// CHECK: /^src{{[/\\]}}top.h
// CHECK: /^tc{{[/\\]}}lib{{[/\\]}}clang{{[/\\]}}{{.*}}{{[/\\]}}include{{[/\\]}}stdarg.h
// CHECK: /^sdk{{[/\\]}}usr{{[/\\]}}include{{[/\\]}}stdlib.h

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
