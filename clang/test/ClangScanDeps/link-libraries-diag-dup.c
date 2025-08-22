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
  }

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

// Note that module D does not report an error because it is explicit.
// Therefore we can use CHECK-NEXT for the redeclaration error on line 15.
// CHECK:      module.modulemap:6:5:  error: link declaration is not allowed in submodules
// CHECK-NEXT: module.modulemap:15:3: error: redeclaration of link library 'libraryA' [-Wmodule-link-redeclaration]
// CHECK-NEXT: module.modulemap:14:3: note: previously declared here
// CHECK-NOT:  module.modulemap:20:3: error: redeclaration of link library 'libraryA'
