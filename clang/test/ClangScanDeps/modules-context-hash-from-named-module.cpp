// Check that the driver-generated command line options for  C++ module inputs
// do not unexpectedly change the context hash of imported Clang modules
// compared to the command line for an otherwise equal non-module input.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- main.cpp
#include "root.h"
import A;
import B;

auto main() -> int { return 1; }

//--- A.cppm
module;
#include "root.h"
export module A;

//--- B.cppm
module;
#include "root.h"
export module B;

//--- module.modulemap
module root { header "root.h" }

//--- root.h
// empty

// RUN: %clang -std=c++23 -fmodules %t/main.cpp %t/A.cppm %t/B.cppm -fsyntax-only -fdriver-only -MJ %t/deps.json
// RUN: sed -e '1s/^/[/' -e '$s/,$/]/' %t/deps.json | sed 's:\\\\\?:/:g' > %t/compile_commands.json
// RUN: clang-scan-deps -compilation-database=%t/compile_commands.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json

// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/module.modulemap",
// CHECK:            "context-hash": "[[HASH_ROOT:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/root.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "root"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "named-module-deps": [
// CHECK-NEXT:             "A",
// CHECK-NEXT:             "B"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_ROOT]]",
// CHECK-NEXT:               "module-name": "root"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/main.cpp"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/main.cpp"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "named-module": "A",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_ROOT]]",
// CHECK-NEXT:               "module-name": "root"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/A.cppm"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/A.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "named-module": "B",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_ROOT]]",
// CHECK-NEXT:               "module-name": "root"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK:                "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/B.cppm"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/B.cppm"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
