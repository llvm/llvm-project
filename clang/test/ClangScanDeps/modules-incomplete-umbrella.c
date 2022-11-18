// This test checks that modules loaded during compilation (but not imported)
// are still reported as dependencies.

// RUN: rm -rf %t && mkdir %t
// RUN: split-file %s %t

//--- frameworks/FW.framework/Modules/module.modulemap
framework module FW {
  umbrella header "FW.h"
  module * { export * }
}
//--- frameworks/FW.framework/Headers/FW.h
//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private {
  umbrella header "FW_Private.h"
  module * { export * }
}
//--- frameworks/FW.framework/PrivateHeaders/FW_Private.h
#include "One.h"
//--- frameworks/FW.framework/PrivateHeaders/One.h
//--- frameworks/FW.framework/PrivateHeaders/Two.h

// Let's check we report the non-imported modular dependencies for a translation unit.

//--- from_tu.cdb.json.template
[{
  "file": "DIR/from_tu.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -F DIR/frameworks -c DIR/from_tu.m -o DIR/from_tu.o"
}]
//--- from_tu.m
#include "FW/FW.h"
#include "FW/Two.h"

// RUN: sed -e "s|DIR|%/t|g" %t/from_tu.cdb.json.template > %t/from_tu.cdb.json
// RUN: clang-scan-deps -compilation-database %t/from_tu.cdb.json -format experimental-full > %t/from_tu_result.json
// RUN: cat %t/from_tu_result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK_TU
// CHECK_TU:      {
// CHECK_TU-NEXT:   "modules": [
// CHECK_TU-NEXT:     {
// CHECK_TU-NEXT:       "clang-module-deps": [],
// CHECK_TU-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK_TU-NEXT:       "command-line": [
// CHECK_TU:            ],
// CHECK_TU-NEXT:       "context-hash": "{{.*}}",
// CHECK_TU-NEXT:       "file-deps": [
// CHECK_TU-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Headers/FW.h",
// CHECK_TU-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap"
// CHECK_TU-NEXT:       ],
// CHECK_TU-NEXT:       "name": "FW"
// CHECK_TU-NEXT:     },
// CHECK_TU-NEXT:     {
// CHECK_TU-NEXT:       "clang-module-deps": [],
// CHECK_TU-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK_TU-NEXT:       "command-line": [
// CHECK_TU:            ],
// CHECK_TU-NEXT:       "context-hash": "{{.*}}",
// CHECK_TU-NEXT:       "file-deps": [
// CHECK_TU-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK_TU-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/FW_Private.h",
// CHECK_TU-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/One.h"
// CHECK_TU-NEXT:       ],
// CHECK_TU-NEXT:       "name": "FW_Private"
// CHECK_TU-NEXT:     }
// CHECK_TU-NEXT:   ],
// CHECK_TU-NEXT:   "translation-units": [
// CHECK_TU-NEXT:     {
// CHECK_TU:            "clang-context-hash": "{{.*}}",
// CHECK_TU-NEXT:       "clang-module-deps": [
// CHECK_TU-NEXT:         {
// CHECK_TU-NEXT:           "context-hash": "{{.*}}",
// CHECK_TU-NEXT:           "module-name": "FW"
// CHECK_TU-NEXT:         },
// CHECK_TU-NEXT:         {
// CHECK_TU-NEXT:           "context-hash": "{{.*}}",
// CHECK_TU-NEXT:           "module-name": "FW_Private"
// CHECK_TU-NEXT:         }
// CHECK_TU-NEXT:       ],
// CHECK_TU-NEXT:       "command-line": [
// CHECK_TU:              "-fmodule-file={{.*}}/FW-{{.*}}.pcm"
// CHECK_TU:              "-fmodule-file={{.*}}/FW_Private-{{.*}}.pcm"
// CHECK_TU:            ],
// CHECK_TU:            "file-deps": [
// CHECK_TU-NEXT:         "[[PREFIX]]/from_tu.m",
// CHECK_TU-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/Two.h"
// CHECK_TU-NEXT:       ],
// CHECK_TU-NEXT:       "input-file": "[[PREFIX]]/from_tu.m"
// CHECK_TU-NEXT:     }

// RUN: %deps-to-rsp %t/from_tu_result.json --module-name=FW > %t/FW.cc1.rsp
// RUN: %deps-to-rsp %t/from_tu_result.json --module-name=FW_Private > %t/FW_Private.cc1.rsp
// RUN: %deps-to-rsp %t/from_tu_result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/FW.cc1.rsp
// RUN: %clang @%t/FW_Private.cc1.rsp
// RUN: %clang @%t/tu.rsp

// Now let's check we report the dependencies for modules as well.

//--- from_module.cdb.json.template
[{
  "file": "DIR/from_module.m",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -F DIR/frameworks -c DIR/from_module.m -o DIR/from_module.o"
}]
//--- module.modulemap
module Mod { header "Mod.h" }
//--- Mod.h
#include "FW/FW.h"
#include "FW/Two.h"
//--- from_module.m
#include "Mod.h"

// RUN: sed -e "s|DIR|%/t|g" %t/from_module.cdb.json.template > %t/from_module.cdb.json
// RUN: clang-scan-deps -compilation-database %t/from_module.cdb.json -format experimental-full > %t/from_module_result.json
// RUN: cat %t/from_module_result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t --check-prefixes=CHECK_MODULE
// CHECK_MODULE:      {
// CHECK_MODULE-NEXT:   "modules": [
// CHECK_MODULE-NEXT:     {
// CHECK_MODULE-NEXT:       "clang-module-deps": [],
// CHECK_MODULE-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK_MODULE-NEXT:       "command-line": [
// CHECK_MODULE:            ],
// CHECK_MODULE-NEXT:       "context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:       "file-deps": [
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Headers/FW.h",
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap"
// CHECK_MODULE-NEXT:       ],
// CHECK_MODULE-NEXT:       "name": "FW"
// CHECK_MODULE-NEXT:     },
// CHECK_MODULE-NEXT:     {
// CHECK_MODULE-NEXT:       "clang-module-deps": [],
// CHECK_MODULE-NEXT:       "clang-modulemap-file": "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK_MODULE-NEXT:       "command-line": [
// CHECK_MODULE:            ],
// CHECK_MODULE-NEXT:       "context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:       "file-deps": [
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/FW_Private.h",
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/One.h"
// CHECK_MODULE-NEXT:       ],
// CHECK_MODULE-NEXT:       "name": "FW_Private"
// CHECK_MODULE-NEXT:     },
// CHECK_MODULE-NEXT:     {
// CHECK_MODULE-NEXT:       "clang-module-deps": [
// CHECK_MODULE-NEXT:         {
// CHECK_MODULE-NEXT:           "context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:           "module-name": "FW"
// CHECK_MODULE-NEXT:         },
// CHECK_MODULE-NEXT:         {
// CHECK_MODULE-NEXT:           "context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:           "module-name": "FW_Private"
// CHECK_MODULE-NEXT:         }
// CHECK_MODULE-NEXT:       ],
// CHECK_MODULE-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK_MODULE-NEXT:       "command-line": [
// CHECK_MODULE:              "-fmodule-file={{.*}}/FW-{{.*}}.pcm"
// CHECK_MODULE:              "-fmodule-file={{.*}}/FW_Private-{{.*}}.pcm"
// CHECK_MODULE:            ],
// CHECK_MODULE-NEXT:       "context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:       "file-deps": [
// CHECK_MODULE-NEXT:         "[[PREFIX]]/Mod.h"
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.modulemap",
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK_MODULE-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/Two.h",
// CHECK_MODULE-NEXT:         "[[PREFIX]]/module.modulemap"
// CHECK_MODULE-NEXT:       ],
// CHECK_MODULE-NEXT:       "name": "Mod"
// CHECK_MODULE-NEXT:     }
// CHECK_MODULE-NEXT:   ],
// CHECK_MODULE-NEXT:   "translation-units": [
// CHECK_MODULE-NEXT:     {
// CHECK_MODULE:            "clang-context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:       "clang-module-deps": [
// CHECK_MODULE-NEXT:         {
// CHECK_MODULE-NEXT:           "context-hash": "{{.*}}",
// CHECK_MODULE-NEXT:           "module-name": "Mod"
// CHECK_MODULE-NEXT:         }
// CHECK_MODULE-NEXT:       ],
// CHECK_MODULE-NEXT:       "command-line": [
// CHECK_MODULE:            ],
// CHECK_MODULE:            "file-deps": [
// CHECK_MODULE-NEXT:         "[[PREFIX]]/from_module.m"
// CHECK_MODULE-NEXT:       ],
// CHECK_MODULE-NEXT:       "input-file": "[[PREFIX]]/from_module.m"
// CHECK_MODULE-NEXT:     }

// RUN: %deps-to-rsp %t/from_module_result.json --module-name=FW > %t/FW.cc1.rsp
// RUN: %deps-to-rsp %t/from_module_result.json --module-name=FW_Private > %t/FW_Private.cc1.rsp
// RUN: %deps-to-rsp %t/from_module_result.json --module-name=Mod > %t/Mod.cc1.rsp
// RUN: %deps-to-rsp %t/from_module_result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/FW.cc1.rsp
// RUN: %clang @%t/FW_Private.cc1.rsp
// RUN: %clang @%t/Mod.cc1.rsp
// RUN: %clang @%t/tu.rsp
