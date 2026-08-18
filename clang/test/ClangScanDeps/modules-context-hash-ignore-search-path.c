// Ensure -fmodules-ignore-search-path drops the path from the module build and
// from its context hash, so a TU that passes such a path shares one module with
// a TU that never mentioned it, while the path itself is still available to the
// TU that did.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -optimize-args=none -format experimental-full -o %t/deps.json

// RUN: ls %t/cache/*/Common-*.pcm | count 1

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/common/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodules-ignore-search-path
// CHECK-NOT:          "[[PREFIX]]/extra"
// CHECK:              "-I"
// CHECK-NEXT:         "[[PREFIX]]/common"
// CHECK-NOT:          "-I"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/common/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/common/common.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Common"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH]]",
// CHECK-NEXT:               "module-name": "Common"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                  "-I",
// CHECK-NEXT:             "[[PREFIX]]/common",
// CHECK:                ],
// CHECK-NEXT:           "executable": "{{.*}}",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "{{.*}}",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH]]",
// CHECK-NEXT:               "module-name": "Common"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                  "-fmodules-ignore-search-path=[[PREFIX]]/extra",
// CHECK-NEXT:             "-I",
// CHECK-NEXT:             "[[PREFIX]]/common",
// CHECK-NEXT:             "-I",
// CHECK-NEXT:             "[[PREFIX]]/extra",
// CHECK:                ],
// CHECK-NEXT:           "executable": "{{.*}}",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT:  }

//--- cdb.json.in
[
  {
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -o DIR/tu1.o -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -I DIR/common",
    "file": "DIR/tu.c"
  },
  {
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -o DIR/tu1.o -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -I DIR/common -I DIR/extra -fmodules-ignore-search-path=DIR/extra",
    "file": "DIR/tu.c"
  }
]

//--- common/module.modulemap
module Common { header "common.h" }
//--- common/common.h

//--- extra/module.modulemap
//--- extra/extra.h

//--- tu.c
#include "common.h"
