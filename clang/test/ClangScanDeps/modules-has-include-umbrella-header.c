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
#include "dependency.h"

//--- modules/module.modulemap
module Poison { header "poison.h" }
module Import { header "import.h" }
module Dependency { header "dependency.h" }
//--- modules/poison.h
#if __has_include(<FW/B.h>)
#define HAS_B 1
#else
#define HAS_B 0
#endif
//--- modules/import.h
#include <FW/B.h>
//--- modules/dependency.h

//--- tu.c
#include "poison.h"

#if __has_include(<FW/B.h>)
#endif

#include "import.h"

#include <FW/B.h>

// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// Let's check that the TU actually depends on `FW_Private` (and does not treat FW/B.h as textual).
// CHECK:      {
// CHECK:        "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "{{.*}}",
// CHECK-NEXT:               "module-name": "FW_Private"
// CHECK-NEXT:             }
// CHECK:                ],
// CHECK-NEXT:           "command-line": [
// CHECK:                ],
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }
