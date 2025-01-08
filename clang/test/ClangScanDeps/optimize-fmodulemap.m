// Check that unused directly passed -fmodule-map-file options get dropped.

// RUN: rm -rf %t && split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/build/cdb.json.in > %t/build/cdb.json
// RUN: clang-scan-deps -compilation-database %t/build/cdb.json \
// RUN:   -format experimental-full > %t/deps.json
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "B"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/A/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/modules/A/module.modulemap"
// CHECK:              "-fmodule-map-file=[[PREFIX]]/modules/B/module.modulemap"
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/modules/A/module.modulemap"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "A"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/modules/B/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodule-map-file=
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK:            ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "B"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK:        ]
// CHECK:      }

//--- build/cdb.json.in
[{
  "directory": "DIR",
  "command": "clang -c DIR/tu.m -I DIR/modules/B -fmodule-map-file=DIR/modules/A/module.modulemap -fmodules -fmodules-cache-path=DIR/cache -fimplicit-module-maps",
  "file": "DIR/tu.m"
}]

//--- build/vfs.yaml.in

//--- tu.m
@import A;

//--- modules/A/module.modulemap
module A { header "A.h" }

//--- modules/A/A.h
#include <B.h>

//--- modules/B/module.modulemap
module B { header "B.h" }

//--- modules/B/B.h
