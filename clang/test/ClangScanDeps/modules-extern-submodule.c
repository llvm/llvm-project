// RUN: rm -rf %t
// RUN: split-file %s %t

//--- tu.m
@import first;

//--- first/first/module.modulemap
module first { header "first.h" }
//--- first/first/first.h
#include <second/sub.h>

//--- second/second/module.modulemap
module second { extern module sub "sub.modulemap" }
//--- second/second/sub.modulemap
module second.sub { header "sub.h" }
//--- second/second/sub.h
@import third;

//--- third/module.modulemap
module third {}

//--- cdb.json.template
[{
  "file": "DIR/tu.c",
  "directory": "DIR",
  "command": "clang -I DIR/first -I DIR/second -I DIR/third -fmodules -fmodules-cache-path=DIR/cache -c DIR/tu.m -o DIR/tu.o"
}]

// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-full > %t/result.json
// RUN: cat %t/result.json | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:       {
// CHECK-NEXT:   "modules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "second"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/first/first/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK:              "-fmodule-map-file=[[PREFIX]]/second/second/module.modulemap"
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/second/second/sub.modulemap"
// CHECK-NOT:          "-fmodule-map-file=[[PREFIX]]/third/module.modulemap"
// CHECK:              "-fmodule-file=second=[[PREFIX]]/cache/{{.*}}/second-{{.*}}.pcm"
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/first/first.h",
// CHECK-NEXT:         "[[PREFIX]]/first/first/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/second/second/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/second/second/sub.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "first"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "third"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/second/second/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK:              "-fmodule-map-file=[[PREFIX]]/third/module.modulemap",
// CHECK:              "-fmodule-file=third=[[PREFIX]]/cache/{{.*}}/third-{{.*}}.pcm",
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/second/second/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/second/second/sub.h",
// CHECK-NEXT:         "[[PREFIX]]/second/second/sub.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/third/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "second"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/third/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK-NEXT:         "-cc1",
// CHECK-NOT:          "-fmodule-map-file=
// CHECK-NOT:          "-fmodule-file=third=
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/third/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "third"
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
// CHECK-NEXT:               "module-name": "first"
// CHECK-NEXT:             }
// CHECK-NEXT:           ],
// CHECK-NEXT:           "command-line": [
// CHECK-NEXT:             "-cc1",
// CHECK:                  "-fmodule-map-file=[[PREFIX]]/first/first/module.modulemap",
// CHECK:                  "-fmodule-file=first=[[PREFIX]]/cache/{{.*}}/first-{{.*}}.pcm",
// CHECK:                ],
// CHECK-NEXT:           "executable": "clang",
// CHECK-NEXT:           "file-deps": [
// CHECK-NEXT:             "[[PREFIX]]/tu.m"
// CHECK-NEXT:           ],
// CHECK-NEXT:           "input-file": "[[PREFIX]]/tu.c"
// CHECK-NEXT:         }
// CHECK:            ]
// CHECK:          }
// CHECK:        ]
// CHECK:      }

// RUN: %deps-to-rsp %t/result.json --module-name=third > %t/third.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=second > %t/second.cc1.rsp
// RUN: %deps-to-rsp %t/result.json --module-name=first > %t/first.cc1.rsp
// RUN: %clang @%t/third.cc1.rsp
// RUN: %clang @%t/second.cc1.rsp
// RUN: %clang @%t/first.cc1.rsp
