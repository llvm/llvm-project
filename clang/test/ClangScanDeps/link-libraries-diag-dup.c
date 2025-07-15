// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

//--- module.modulemap
module A {
  umbrella header "A.h"

  link "libraryA"
  link "libraryA"
}

//--- A.h
// empty
//--- TU.c
#include "A.h"

//--- cdb.json.template
[{
  "file": "DIR/TU.c",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR -c DIR/TU.c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: not clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full 2>&1 | FileCheck %s

// CHECK:      module.modulemap:5:3: error: redeclaration of link library 'libraryA'
// CHECK-NEXT: module.modulemap:4:3: note: previously declared here
