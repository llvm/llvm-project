// Ensure '-DFOO -fmodules-ignore-macro=FOO' and '' both produce the same
// canonical module build command.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK:          {
// CHECK:            "command-line": [
// CHECK-NOT:          "FOO"
// CHECK-NOT:          "-fmodules-ignore-macro
// CHECK:            ]
// CHECK:            "context-hash": "[[HASH_NO_FOO:.*]]"
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK:            "command-line": [
// CHECK:              "-D"
// CHECK-NEXT:         "FOO"
// CHECK:            ]
// CHECK:            "context-hash": "[[HASH_FOO:.*]]"
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_NO_FOO]]"
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-DFOO"
// CHECK:            ]
// CHECK:            "input-file": "{{.*}}tu1.c"
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_FOO]]"
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:       "command-line": [
// CHECK:              "-DFOO"
// CHECK:            ]
// CHECK:            "input-file": "{{.*}}tu2.c"
// CHECK-NEXT:     }
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_NO_FOO]]"
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:       "command-line": [
// CHECK:              "-DFOO"
// CHECK:              "-fmodules-ignore-macro=FOO"
// CHECK:            ]
// CHECK:            "input-file": "{{.*}}tu3.c"

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu1.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu1.c"
  },
  {
    "directory": "DIR",
    "command": "clang -DFOO -fsyntax-only DIR/tu2.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu2.c"
  },
  {
    "directory": "DIR",
    "command": "clang -DFOO -fmodules-ignore-macro=FOO -fsyntax-only DIR/tu3.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu3.c"
  },
]

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu1.c
#include "Mod.h"

//--- tu2.c
#include "Mod.h"

//--- tu3.c
#include "Mod.h"
