// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t

//--- module.modulemap
module A {
  umbrella header "A.h"

  module B {
    header "B.h"
    link "libraryB"
  }

  explicit module D {
    header "D.h"
    link "libraryD"
    link "libraryD"
  }

  link "libraryB"
  link "libraryA"
  link "libraryA"
}

module C {
  header "C.h"
  link "libraryA"
}

//--- A.h
#include "B.h"
//--- B.h
// empty
//--- C.h
// empty
//--- D.h
// empty
//--- TU.c
#include "A.h"
#include "C.h"
#include "D.h"

//--- cdb.json.template
[{
  "file": "DIR/TU.c",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR -c DIR/TU.c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: not clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full 2>&1 | FileCheck %s

// Note that the `link "libraryB"` in the top level module A does not
// cause an issue because we only check within a module.
// CHECK: 12:5: error: redeclaration of link library 'libraryD' [-Wmodule-link-redeclaration]
// CHECK-NEXT: 11:5: note: previously declared here
// CHECK-NEXT: 17:3: error: redeclaration of link library 'libraryA' [-Wmodule-link-redeclaration]
// CHECK-NEXT: 16:3: note: previously declared here
// CHECK-NOT: error: redeclaration of link library 'libraryB'
