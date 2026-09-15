// Test that scanning multiple modules by name does not crash due to
// use-after-free of PreprocessingRecord when removePPCallbacks() is
// called between scans. The option -detailed-preprocessing-record leads
// to the creation of a PreprocessingRecord instance.
// 
// Without the corresponding fix in the Preprocessor, this test fails with
// address sanitizer.
//
// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module A {
  header "A.h"
}
module B {
  header "B.h"
  use A
}

//--- A.h
#define A_VALUE 1

//--- B.h
#include "A.h"
int b_val = A_VALUE;

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fimplicit-module-maps -IDIR -Xclang -detailed-preprocessing-record -x c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full -module-names=A,B > %t/result.json
// RUN: cat %t/result.json | FileCheck %s

// CHECK:      "modules": [
// CHECK:          "name": "A"
// CHECK:          "name": "B"
