// This test checks that inferred frameworks/modules are accounted for in the
// scanner and can be explicitly built by Clang.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/Inferred.framework/Headers/Inferred.h
typedef int inferred;

//--- frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h

//--- frameworks/module.modulemap
framework module * {}

//--- tu.m
#include <Inferred/Inferred.h>

inferred a = 0;

//--- cdb.json.template
[{
  "directory": "DIR",
  "file": "DIR/tu.m",
  "command": "clang -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -F DIR/frameworks -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/Inferred.framework/Headers/Inferred.h",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/Inferred.framework/Frameworks/Sub.framework/Headers/Sub.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "isFramework": true,
// CHECK-NEXT:           "link-name": "Inferred"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "Inferred"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "{{.*}}",
// CHECK-NEXT:               "module-name": "Inferred"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                ],
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }

// RUN: %deps-to-rsp %t/result.json --module-name=Inferred > %t/Inferred.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/Inferred.cc1.rsp -pedantic -Werror
// RUN: %clang @%t/tu.rsp -pedantic -Werror
