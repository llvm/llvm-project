// Check that we canonicalize the module map path without changing the module
// directory, which would break header lookup.

// REQUIRES: shell

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json
// RUN: ln -s module %t/symlink-to-module
// RUN: ln -s ../actual.modulemap %t/module/module.modulemap
// RUN: ln -s A %t/module/F.framework/Versions/Current
// RUN: ln -s Versions/Current/Modules %t/module/F.framework/Modules
// RUN: ln -s Versions/Current/Headers %t/module/F.framework/Headers

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -format experimental-full  -mode=preprocess-dependency-directives \
// RUN:   -optimize-args -module-files-dir %t/build > %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// Check the module commands actually build.
// RUN: %deps-to-rsp %t/deps.json --module-name=Mod > %t/Mod.rsp
// RUN: %deps-to-rsp %t/deps.json --module-name=F > %t/F.rsp
// RUN: %clang @%t/Mod.rsp
// RUN: %clang @%t/F.rsp

// CHECK:      "modules": [
// CHECK:        {
// CHECK:          "clang-module-deps": [],
// CHECK:          "clang-modulemap-file": "[[PREFIX]]/module/F.framework/Modules/module.modulemap"
// CHECK:          "command-line": [
// CHECK-NOT: symlink-to-module
// CHECK:            "[[PREFIX]]/module/F.framework/Modules/module.modulemap"
// CHECK-NOT: symlink-to-module
// CHECK:          ]
// CHECK:          "context-hash": "[[F_CONTEXT_HASH:[A-Z0-9]+]]"
// CHECK:          "name": "F"
// CHECK-NEXT:   }
// CHECK-NEXT:   {
// CHECK:          "clang-modulemap-file": "[[PREFIX]]/module/module.modulemap"
// CHECK:          "command-line": [
// CHECK-NOT: symlink-to-module
// CHECK:            "[[PREFIX]]/module/module.modulemap"
// CHECK-NOT: symlink-to-module
// CHECK:          ]
// CHECK:          "context-hash": "[[CONTEXT_HASH:[A-Z0-9]+]]"
// CHECK:          "name": "Mod"
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:      "translation-units": [
// CHECK:              "clang-module-deps": [
// CHECK:                {
// CHECK:                  "context-hash": "[[CONTEXT_HASH]]"
// CHECK:                  "module-name": "Mod"
// CHECK:                }
// CHECK-NEXT:         ],
// CHECK:              "command-line": [
// CHECK:                "-fmodule-map-file=[[PREFIX]]/module/module.modulemap"
// CHECK:              ]
// CHECK:              "clang-module-deps": [
// CHECK:                {
// CHECK:                  "context-hash": "[[CONTEXT_HASH]]"
// CHECK:                  "module-name": "Mod"
// CHECK:                }
// CHECK-NEXT:         ]
// CHECK:              "command-line": [
// CHECK:                "-fmodule-map-file=[[PREFIX]]/module/module.modulemap"
// CHECK:              ]
// CHECK:              "clang-module-deps": [
// CHECK:                {
// CHECK:                  "context-hash": "[[F_CONTEXT_HASH]]"
// CHECK:                  "module-name": "F"
// CHECK:                }
// CHECK-NEXT:         ]
// CHECK:              "command-line": [
// CHECK:                "-fmodule-map-file=[[PREFIX]]/module/F.framework/Modules/module.modulemap"
// CHECK:              ]
// CHECK:              "clang-module-deps": [
// CHECK:                {
// CHECK:                  "context-hash": "[[F_CONTEXT_HASH]]"
// CHECK:                  "module-name": "F"
// CHECK:                }
// CHECK-NEXT:         ]
// CHECK:              "command-line": [
// CHECK:                "-fmodule-map-file=[[PREFIX]]/module/F.framework/Modules/module.modulemap"
// CHECK:              ]

//--- cdb.json.in
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu1.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu1.c"
  },
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/tu2.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu2.c"
  },
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only -F DIR/symlink-to-module DIR/tu3.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu3.c"
  },
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only -F DIR/module DIR/tu3.c -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache",
    "file": "DIR/tu3.c"
  },
]

//--- actual.modulemap
module Mod { header "header.h" }

//--- module/header.h

//--- tu1.c
#include "symlink-to-module/header.h"

//--- tu2.c
#include "module/header.h"

//--- module/F.framework/Versions/A/Modules/module.modulemap
framework module F {
  umbrella header "F.h"
}

//--- module/F.framework/Versions/A/Headers/F.h

//--- tu3.c
#include "F/F.h"
