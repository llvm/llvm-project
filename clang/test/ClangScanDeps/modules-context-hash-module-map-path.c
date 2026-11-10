// Ensure the path to the modulemap input is included in the context hash
// irrespective of other TU command-line arguments, as it effects the canonical
// module build command. In this test we use the difference in case of a path.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/overlay.json.template > %t/overlay.json

// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -I %t/dir -I %t/Dir -c %t/tu0.c -ivfsoverlay %t/overlay.json \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   > %t/deps.json

// RUN: echo 'DIFFERENT_PATH' >> %t/deps.json

// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -I %t/dir -I %t/Dir -c %t/tu1.c -ivfsoverlay %t/overlay.json \
// RUN:     -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   >> %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK:          {
// CHECK:            "command-line": [
// CHECK:              "{{.*}}Dir/module.modulemap"
// CHECK:            ]
// CHECK:            "context-hash": "[[HASH1:.*]]"
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH1]]"
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-LABEL: DIFFERENT_PATH
// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK:          {
// CHECK-NOT: [[HASH1]]
// CHECK:            "command-line": [
// CHECK:              "{{.*}}dir/module.modulemap"
// CHECK:            ]
// CHECK-NOT: [[HASH1]]
// CHECK:            "name": "Mod"
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK:            "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash":
// CHECK-NOT: [[HASH1]]
// CHECK-NEXT:            "module-name": "Mod"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]

// This overlay is used just to force the filesystem to be case sensitive to
// simulate case insensitive access via diffierent capitalizations.
//--- overlay.json.template

{
  "version": 0,
  "case-sensitive": true,
  "roots": [
  {
     "contents": [
     {
        "external-contents": "DIR/m.m",
        "name": "module.modulemap",
        "type": "file"
     },
     {
        "external-contents": "DIR/Mod.h",
        "name": "Mod.h",
        "type": "file"
     }],
     "name": "DIR/Dir",
     "type": "directory"
  },
  {
     "contents": [
     {
        "external-contents": "DIR/m.m",
        "name": "module.modulemap",
        "type": "file"
     },
     {
        "external-contents": "DIR/Mod.h",
        "name": "Mod.h",
        "type": "file"
     }],
     "name": "DIR/dir",
     "type": "directory"
  }
  ]
}

//--- m.m
module Mod { header "Mod.h" }

//--- Mod.h

//--- tu0.c
#include "Dir/Mod.h"

//--- tu1.c
#include "dir/Mod.h"
