// RUN: rm -rf %t
// RUN: split-file %s %t

//--- tu.m
#include "first.h"

//--- first/module.modulemap
module first { header "first.h" }
//--- first/first.h
#include "second.h"

//--- second/module.modulemap
module second { header "second.h" }
//--- second/second.h
#include "third.h"

//--- third/module.modulemap
module third { header "third.h" }
//--- third/third.h
// empty

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -I DIR/first -I DIR/second -I DIR/third -fmodules -fmodules-cache-path=DIR/cache -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "second"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/third/module.modulemap"
// CHECK:              "-fmodule-map-file=[[PREFIX]]/second/module.modulemap"
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/third/module.modulemap"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/first/first.h",
// CHECK-NEXT:         "[[PREFIX]]/second/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "first"
// CHECK-NEXT:     }
// CHECK:        ]
// CHECK:      }
