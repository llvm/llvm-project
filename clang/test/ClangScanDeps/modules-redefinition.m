// This test checks that we don't report non-affecting system module maps.
// More specifically, we check that explicitly-specified module map file is not
// being shadowed in explicit build by a module map file found during implicit
// module map search.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- tu.m
@import first;

//--- zeroth/module.modulemap
module X {}

//--- first/module.modulemap
module first { header "first.h" }
//--- first/first.h
@import third;
//--- second/module.modulemap
module X {}
//--- third/module.modulemap
module third {}

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fmodule-name=X -fmodule-map-file=DIR/zeroth/module.modulemap -isystem DIR/first -isystem DIR/second -isystem DIR/third -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "third"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         [[PREFIX]]/first/first.h",
// CHECK-NEXT:         [[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:         [[PREFIX]]/third/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "first"
// CHECK-NEXT:     }
// CHECK:        ]
// CHECK:      }

// RUN: %deps-to-rsp %t/result.json --module-name=third > %t/third.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=first > %t/first.cc1.rsp
// RUN: %clang @%t/third.cc1.rsp
// RUN: %clang @%t/first.cc1.rsp
