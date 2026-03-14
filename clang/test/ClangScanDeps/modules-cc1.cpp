// Check that clang-scan-deps works with cc1 command lines

// RUN: rm -rf %t
// RUN: split-file %s %t


//--- modules_cc1.cpp
#include "header.h"

//--- header.h

//--- module.modulemap
module header1 { header "header.h" }

//--- cdb.json.template
[{
  "file": "DIR/modules_cc1.cpp",
  "directory": "DIR",
  "command": "clang -cc1 DIR/modules_cc1.cpp -fimplicit-module-maps -o modules_cc1.o"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 -mode preprocess-dependency-directives > %t/result
// RUN: cat %t/result | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK: modules_cc1.o:
// CHECK-NEXT: [[PREFIX]]/modules_cc1.cpp
// CHECK-NEXT: [[PREFIX]]/module.modulemap
// CHECK-NEXT: [[PREFIX]]/header.h
