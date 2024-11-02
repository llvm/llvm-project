// Unsupported on AIX because we don't support the requisite "__clangast"
// section in XCOFF yet.
// UNSUPPORTED: aix

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- undeclared/module.modulemap
module Undeclared { header "undeclared.h" }

//--- undeclared/undeclared.h

//--- module.modulemap
module User [no_undeclared_includes] { header "user.h" }

//--- user.h
#if __has_include("undeclared.h")
#error Unreachable. Undeclared comes from a module that's not 'use'd, meaning the compiler should pretend it doesn't exist.
#endif

//--- test.c
#include "user.h"

//--- cdb.json.template
[{
  "directory": "DIR",
  "command": "clang -fmodules -gmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -IDIR/undeclared -c DIR/test.c -o DIR/test.o",
  "file": "DIR/test.c"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full \
// RUN:   -module-files-dir %t/build > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:        {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fmodule-map-file=[[PREFIX]]/undeclared/module.modulemap"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/undeclared/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/user.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "User"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "{{.*}}"
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}"
// CHECK-NEXT:           "module-name": "User"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/test.c"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/test.c"
// CHECK-NEXT:     }

// RUN: %deps-to-rsp %t/result.json --module-name=User > %t/User.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
//
// RUN: %clang @%t/User.cc1.rsp
// RUN: %clang @%t/tu.rsp
