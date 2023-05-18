// This test checks that we're reporting module maps affecting the compilation by describing excluded headers.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/X.framework/Modules/module.modulemap
framework module X {
  umbrella header "X.h"
  exclude header "Excluded.h"
}
//--- frameworks/X.framework/Headers/X.h
//--- frameworks/X.framework/Headers/Excluded.h

//--- mod/module.modulemap
module Mod { header "Mod.h" }
//--- mod/Mod.h
#include <X/Excluded.h>

//--- tu.m
@import Mod;

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR/mod -F DIR/frameworks -Werror=non-modular-include-in-module -c DIR/tu.m -o DIR/tu.m"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/mod/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fmodule-map-file=[[PREFIX]]/frameworks/X.framework/Modules/module.modulemap"
// CHECK-NOT:          "-fmodule-file={{.*}}"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK:      }

// RUN: %deps-to-rsp %t/result.json --module-name=Mod > %t/Mod.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp

// RUN: %clang @%t/Mod.cc1.rsp
// RUN: %clang @%t/tu.rsp
