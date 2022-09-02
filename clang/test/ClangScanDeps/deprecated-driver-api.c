// Test the deprecated version of the API that returns a driver command instead
// of multiple -cc1 commands.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// RUN: clang-scan-deps -compilation-database=%t/cdb.json -format experimental-full \
// RUN:   -deprecated-driver-command | sed 's:\\\\\?:/:g' | FileCheck %s

// CHECK: "command-line": [
// CHECK:   "-c"
// CHECK:   "{{.*}}tu.c"
// CHECK:   "-save-temps"
// CHECK:   "-fno-implicit-modules"
// CHECK:   "-fno-implicit-module-maps"
// CHECK: ]
// CHECK: "file-deps": [
// CHECK:   "{{.*}}tu.c",
// CHECK:   "{{.*}}header.h"
// CHECK: ]

//--- cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.c -save-temps",
  "file": "DIR/tu.c"
}]

//--- header.h
void bar(void);

//--- tu.c
#include "header.h"

void foo(void) {
  bar();
}
