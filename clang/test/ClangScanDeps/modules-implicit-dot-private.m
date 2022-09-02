// This test checks that modules loaded during compilation (but not imported)
// are still reported as dependencies.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW { umbrella header "FW.h" }
//--- frameworks/FW.framework/Headers/FW.h
//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private { umbrella header "FW_Private.h" }
//--- frameworks/FW.framework/PrivateHeaders/FW_Private.h

//--- cdb.json.template
[{
  "file": "DIR/tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -iframework DIR/frameworks -c DIR/tu.m -o DIR/tu.o"
}]
//--- tu.m
@import FW.Private;

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Headers/FW.h",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "FW"
// CHECK:          },
// CHECK:          {
// CHECK:            "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/FW_Private.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "FW_Private"
// CHECK:           }
// CHECK:        ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-context-hash": "{{.*}}",
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "FW"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "FW_Private"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fmodule-file={{.*}}/FW-{{.*}}.pcm"
// CHECK:              "-fmodule-file={{.*}}/FW_Private-{{.*}}.pcm"
// CHECK:            ],
// CHECK:            "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/tu.m"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:     }

// RUN: %deps-to-rsp %t/result.json --module-name=FW > %t/FW.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=FW_Private > %t/FW_Private.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/FW.cc1.rsp
// RUN: %clang @%t/FW_Private.cc1.rsp
// RUN: %clang @%t/tu.rsp
