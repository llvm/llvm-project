// Check that -fmodules-ignore-search-path composes with the scanner's own
// usage-based pruning of header search paths (-optimize-args=header-search).

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.in > %t/cdb.json

// The ignored path is the *first* -I, so dropping it shifts the index of every
// path the modules do use.
//
// RUN: clang-scan-deps -compilation-database %t/cdb.json -j 1 \
// RUN:   -optimize-args=header-search -format experimental-full -o %t/deps.json

// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/leaf/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodules-ignore-search-path
// CHECK-NOT:          "-I"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_LEAF:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/leaf/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/leaf/leaf.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Leaf"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_LEAF]]",
// CHECK-NEXT:           "module-name": "Leaf"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/middle/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodules-ignore-search-path
// CHECK:              "-I"
// CHECK-NEXT:         "[[PREFIX]]/leaf"
// CHECK-NOT:          "-I"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_MIDDLE:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/middle/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/middle/middle.h",
// CHECK-NEXT:         "[[PREFIX]]/leaf/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Middle"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "[[HASH_MIDDLE]]",
// CHECK-NEXT:           "module-name": "Middle"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/top/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodules-ignore-search-path
// CHECK:              "-I"
// CHECK-NEXT:         "[[PREFIX]]/middle"
// CHECK-NEXT:         "-I"
// CHECK-NEXT:         "[[PREFIX]]/leaf"
// CHECK-NOT:          "-I"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "[[HASH_TOP:.*]]",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/top/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/top/top.h",
// CHECK-NEXT:         "[[PREFIX]]/middle/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "Top"
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "translation-units": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "[[HASH_TU:.*]]",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_TOP]]",
// CHECK-NEXT:               "module-name": "Top"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                 "-fmodules-ignore-search-path=[[PREFIX]]/ignored1"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/ignored1"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/top"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/middle"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/leaf"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/unused"
// CHECK:                ],
// CHECK-NEXT:           "executable": "{{.*}}",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c",
// CHECK-NEXT:             "[[PREFIX]]/ignored1/ignored.h"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "commands": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "clang-context-hash": "[[HASH_TU]]",
// CHECK-NEXT:           "clang-module-deps": [
// CHECK-NEXT:             {
// CHECK-NEXT:               "context-hash": "[[HASH_TOP]]",
// CHECK-NEXT:               "module-name": "Top"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                  "-fmodules-ignore-search-path=[[PREFIX]]/ignored2"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/ignored2"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/top"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/middle"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/leaf"
// CHECK-NEXT:             "-I"
// CHECK-NEXT:             "[[PREFIX]]/unused"
// CHECK:                ],
// CHECK-NEXT:           "executable": "{{.*}}",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.c",
// CHECK-NEXT:             "[[PREFIX]]/ignored2/ignored.h"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }

//--- cdb.json.in
[
  {
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -o DIR/tu1.o -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -I DIR/ignored1 -I DIR/top -I DIR/middle -I DIR/leaf -I DIR/unused -fmodules-ignore-search-path=DIR/ignored1",
    "file": "DIR/tu.c"
  },
  {
    "directory": "DIR",
    "command": "clang -c DIR/tu.c -o DIR/tu2.o -fmodules -fimplicit-module-maps -fmodules-cache-path=DIR/cache -I DIR/ignored2 -I DIR/top -I DIR/middle -I DIR/leaf -I DIR/unused -fmodules-ignore-search-path=DIR/ignored2",
    "file": "DIR/tu.c"
  }
]

//--- top/module.modulemap
module Top { header "top.h" }

//--- top/top.h
#include "middle.h"

//--- middle/module.modulemap
module Middle { header "middle.h" }

//--- middle/middle.h
#include "leaf.h"

//--- leaf/module.modulemap
module Leaf { header "leaf.h" }

//--- leaf/leaf.h
int leaf(void);

//--- unused/unused.h
int unused(void);

//--- ignored1/ignored.h
int ignored(void);

//--- ignored2/ignored.h
int ignored(void);

//--- tu.c
#include "top.h"
#include "ignored.h"
