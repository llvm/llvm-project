// The slash direction in linux and windows are different.
// Also the command to create symbolic link is different.
// UNSUPPORTED: system-windows
//
// RUN: rm -fr %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: EXPECTED_RESOURCE_DIR=`%clang -print-resource-dir` && \
// RUN: ln -s %clang++ %t/clang++ && \
// RUN: sed "s|EXPECTED_RESOURCE_DIR|$EXPECTED_RESOURCE_DIR|g; s|DIR|%/t|g" %t/P1689.json.in > %t/P1689.json && \
// RUN: clang-scan-deps -compilation-database %t/P1689.json -format=p1689 | FileCheck %t/a.cpp -DPREFIX=%/t && \
// RUN: clang-scan-deps -format=p1689 \
// RUN:   -- %t/clang++ -std=c++20 -c -fprebuilt-module-path=%t %t/a.cpp -o %t/a.o \
// RUN:      -resource-dir $EXPECTED_RESOURCE_DIR | FileCheck %t/a.cpp -DPREFIX=%/t

//--- P1689.json.in
[
{
  "directory": "DIR",
  "command": "DIR/clang++ -std=c++20 -c -fprebuilt-module-path=DIR DIR/a.cpp -o DIR/a.o -resource-dir EXPECTED_RESOURCE_DIR",
  "file": "DIR/a.cpp",
  "output": "DIR/a.o"
}
]

//--- a.cpp
#include "a.h"
import b;

// CHECK: {
// CHECK-NEXT:   "revision": 0,
// CHECK-NEXT:   "rules": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "primary-output": "[[PREFIX]]/a.o",
// CHECK-NEXT:       "requires": [
// CHECK-NEXT:         {
// CHECK-NEXT:           "logical-name": "b"
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "version": 1
// CHECK-NEXT: }

//--- a.h
