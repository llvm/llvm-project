// This test checks that only module map files defining affecting modules are
// affecting.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- tu.m
@import zeroth;

//--- zeroth/module.modulemap
module zeroth { header "zeroth.h" }
//--- zeroth/zeroth.h
@import first;
#include "second.h"

//--- first/module.modulemap
module first {}
module first_other { header "first_other.h" }
//--- first/first_other.h

//--- second/module.modulemap
extern module second "second.modulemap"
//--- second/second.modulemap
module second { header "second.h" }
//--- second/second.h
#include "first_other.h"

// RUN: clang-scan-deps -format experimental-full -o %t/result.json \
// RUN:   -- %clang -fmodules -fmodules-cache-path=%t/cache -I %t/zeroth -I %t/first -I %t/second -c %t/tu.m -o %t/tu.o
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "first"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/first_other.h",
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "first_other"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "first_other"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/second/second.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/second/module.modulemap"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/second/second.h",
// CHECK-NEXT:         "[[PREFIX]]/second/second.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "second"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "first"
// CHECK-NEXT:         },
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "second"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/zeroth/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/second/module.modulemap"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/second/second.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/zeroth/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/zeroth/zeroth.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "zeroth"
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
// CHECK-NEXT:               "module-name": "zeroth"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK:                ],
// CHECK-NEXT:           "executable": "{{.*}}",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.m"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }

// RUN: %deps-to-rsp --module-name=first %t/result.json > %t/first.cc1.rsp
// RUN: %deps-to-rsp --module-name=first_other %t/result.json > %t/first_other.cc1.rsp
// RUN: %deps-to-rsp --module-name=second %t/result.json > %t/second.cc1.rsp
// RUN: %deps-to-rsp --module-name=zeroth %t/result.json > %t/zeroth.cc1.rsp
// RUN: %clang @%t/first.cc1.rsp
// RUN: %clang @%t/first_other.cc1.rsp
// RUN: %clang @%t/second.cc1.rsp
// RUN: %clang @%t/zeroth.cc1.rsp
