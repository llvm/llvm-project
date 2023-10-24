// This test checks that __has_include(<FW/PrivateHeader.h>) in a module does
// not clobber #include <FW/PrivateHeader.h> in importers of said module.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang DIR/tu.c -fmodules -fmodules-cache-path=DIR/cache -I DIR/modules -F DIR/frameworks -o DIR/tu.o"
}]

//--- frameworks/FW.framework/Modules/module.private.modulemap
framework module FW_Private {
  umbrella header "A.h"
  module * { export * }
}
//--- frameworks/FW.framework/PrivateHeaders/A.h
#include <FW/B.h>
//--- frameworks/FW.framework/PrivateHeaders/B.h
struct B {};

//--- modules/module.modulemap
module Foo { header "foo.h" }
//--- modules/foo.h
#if __has_include(<FW/B.h>)
#define HAS_B 1
#else
#define HAS_B 0
#endif

//--- tu.c
#include "foo.h"
#include <FW/B.h>

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%t

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
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/A.h",
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/PrivateHeaders/B.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "FW_Private"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:              "-fmodule-map-file=[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// FIXME: The frameworks/FW.framework/PrivateHeaders/B.h header never makes it into SourceManager,
//        so we don't track it as a file dependency (even though we should).
// CHECK-NEXT:         "[[PREFIX]]/frameworks/FW.framework/Modules/module.private.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/modules/foo.h",
// CHECK-NEXT:         "[[PREFIX]]/modules/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "name": "Foo"
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
// CHECK-NEXT:             },
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "{{.*}}",
// CHECK-NEXT:               "module-name": "Foo"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK:              }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }
