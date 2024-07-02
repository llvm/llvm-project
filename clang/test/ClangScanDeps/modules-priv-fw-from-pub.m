// This test checks that importing private headers from the public headers of
// a framework is consistent between the dependency scanner and the explicit build.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { header "FW.h" }
//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private { header "FW_Private.h"}
//--- frameworks/FW.framework/Headers/FW.h
#include <FW/FW_Private.h>
//--- frameworks/FW.framework/PrivateHeaders/FW_Private.h
@import Dependency;

//--- modules/module.modulemap
module Dependency { header "dependency.h" }
//--- modules/dependency.h
// empty

//--- tu.m
#include <FW/FW.h>

//--- cdb.json.in
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -fimplicit-module-maps -I DIR/modules -F DIR/frameworks -Wno-framework-include-private-from-public -Wno-atimport-in-framework-header -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/deps.json

// Check that FW is reported to depend on FW_Private.
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/modules/dependency.h",
// CHECK-NEXT:         "[[PREFIX]]/modules/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Dependency"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "FW_Private"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Headers/FW.h",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "isFramework": true,
// CHECK-NEXT:           "link-name": "FW"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "FW"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "Dependency"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/FW_Private.h",
// CHECK-NEXT:         "[[PREFIX]]/modules/module.modulemap"
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
// CHECK-NEXT:               "module-name": "FW"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

// Check that building FW succeeds. If FW_Private was to be treated textually,
// building FW would fail due to Dependency not being present on the command line.
// RUN: %deps-to-rsp %t/deps.json --module-name=Dependency > %t/Dependency.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=FW_Private > %t/FW_Private.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=FW > %t/FW.cc1.rsp
// RUN: %deps-to-rsp %t/deps.json --tu-index=0 > %t/tu.rsp

// RUN: %clang @%t/Dependency.cc1.rsp
// RUN: %clang @%t/FW_Private.cc1.rsp
// RUN: %clang @%t/FW.cc1.rsp
// RUN: %clang @%t/tu.rsp
