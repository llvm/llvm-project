// This test checks that we don't crash or report spurious dependencies on
// FW_Private when compiling the implementation of framework module FW.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "directory": "DIR",
  "file": "DIR/tu.m",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fmodule-name=FW -F DIR/frameworks -c DIR/tu.m -o DIR/tu.o"
}]

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { umbrella header "FW.h" }
//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private { umbrella header "FW_Private.h" }
//--- frameworks/FW.framework/Headers/FW.h
//--- frameworks/FW.framework/PrivateHeaders/FW_Private.h
//--- frameworks/FW.framework/PrivateHeaders/Missed.h
#import <FW/FW.h> // When included from tu.m, this ends up adding (spurious) dependency on FW for FW_Private.

//--- tu.m
@import FW_Private; // This is a direct dependency.
#import <FW/Missed.h>

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/FW_Private.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "isFramework": true,
// CHECK-NEXT:           "link-name": "FW"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "FW_Private"
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
// CHECK-NEXT:               "module-name": "FW_Private"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m",
// CHECK-NEXT:             "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/Missed.h",
// CHECK-NEXT:             "[[PREFIX]]/frameworks/FW.framework/Headers/FW.h"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }
