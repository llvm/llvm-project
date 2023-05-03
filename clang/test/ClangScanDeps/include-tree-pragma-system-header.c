// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -cas-path %t/cas -format experimental-include-tree-full \
// RUN:   -compilation-database %t/cdb.json > %t/deps.json
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp

// Confirm we match the non-include-tree build:
// RUN: %clang -fsyntax-only -Wextra-semi %t/tu.c 2>&1 | FileCheck %s
// RUN: %clang @%t/tu.rsp 2>&1 | FileCheck %s

// CHECK: sys.h:1:7: warning: extra ';'
// CHECK-NOT: warning: extra ';'
// CHECK: tu.c:{{.*}}: warning: extra ';'

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fsyntax-only -Wextra-semi DIR/tu.c",
  "file": "DIR/tu.c"
}]

//--- sys.h
int x;;
#pragma clang system_header
int y;;
#include "other.h"

//--- other.h
int z;;

//--- tu.c
#include "sys.h"
int w;;

int main() {
  return x + y + z;
}