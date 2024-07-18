// This test checks that import directives with missing filenames are ignored when scanning but will result 
// in compile time errors if they need to be parsed.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- tu.m
#import "zeroth.h" 

//--- zeroth/module.modulemap
module zeroth { header "zeroth.h" }
//--- zeroth/zeroth.h
#ifdef BAD_IMPORT 
@import;
#import
#endif 
@import first;

//--- first/module.modulemap
module first { header "first.h" }
//--- first/first.h

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
// CHECK-NEXT:         "[[PREFIX]]/first/first.h",
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "first"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "clang-module-deps": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "context-hash": "{{.*}}",
// CHECK-NEXT:           "module-name": "first"
// CHECK-NEXT:         }
// CHECK-NEXT:       ],
// CHECK-NEXT:       "clang-modulemap-file": "[[PREFIX]]/zeroth/module.modulemap",
// CHECK-NEXT:       "command-line": [
// CHECK:            ],
// CHECK-NEXT:       "context-hash": "{{.*}}",
// CHECK-NEXT:       "file-deps": [
// CHECK-NEXT:         "[[PREFIX]]/first/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/zeroth/module.modulemap",
// CHECK-NEXT:         "[[PREFIX]]/zeroth/zeroth.h"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "link-libraries": [],
// CHECK-NEXT:       "name": "zeroth"
// CHECK-NEXT:     }
// CHECK-NEXT: ], 
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
// RUN: %deps-to-rsp --module-name=zeroth %t/result.json > %t/zeroth.cc1.rsp
// RUN: %clang @%t/first.cc1.rsp
// RUN: %clang @%t/zeroth.cc1.rsp

// Validate that invalid directive that will be parsed results clang error.
// RUN: not clang-scan-deps -format experimental-full -o %t/result_with_bad_imports.json \
// RUN:   -- %clang -fmodules -fmodules-cache-path=%t/diff_cache -I %t/zeroth -I %t/first \
// RUN:      -I %t/second -c %t/tu.m -o %t/tu.o -DBAD_IMPORT=1  2>&1 | FileCheck %s --check-prefix=BAD_IMPORT

// BAD_IMPORT: error: expected a module name after 'import'
// BAD_IMPORT: error: expected "FILENAME" or <FILENAME>

