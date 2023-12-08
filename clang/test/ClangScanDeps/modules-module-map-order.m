// RUN: rm -rf %t
// RUN: split-file %s %t

//--- this/module.modulemap
module This { header "This.h" }
//--- this/This.h
#include "Foo.h"
#include "Foo_Private_Excluded.h"

//--- modules/module.modulemap
module Foo { header "Foo.h" }
//--- modules/module.private.modulemap
explicit module Foo.Private {
  header "Foo_Private.h"
  exclude header "Foo_Private_Excluded.h"
}
//--- modules/Foo.h
//--- modules/Foo_Private.h
//--- modules/Foo_Private_Excluded.h

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR/this -I DIR/modules -c DIR/tu.m -o DIR/tu.o"
}]

//--- tu.m
@import This;

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json

// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:          },
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK:              "-fmodule-map-file=[[PREFIX]]/modules/module.modulemap",
// CHECK-NEXT:         "-fmodule-map-file=[[PREFIX]]/modules/module.private.modulemap",
// CHECK:            ],
// CHECK:            "name": "This"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK:      }

// RUN: %deps-to-rsp %t/result.json --module-name=Foo > %t/Foo.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=This > %t/This.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp

// RUN: %clang @%t/Foo.cc1.rsp
// RUN: %clang @%t/This.cc1.rsp
// RUN: %clang @%t/tu.rsp
