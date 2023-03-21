// RUN: rm -rf %t
// RUN: split-file --leading-lines %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -cas-path %t/cas > %t/result1.txt
// Try again to ensure a pre-populated CASDB doesn't change output.
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -cas-path %t/cas > %t/result2.txt
// RUN: diff -u %t/result1.txt %t/result2.txt
// RUN: FileCheck %s -input-file %t/result1.txt -DPREFIX=%/t

// Make sure order is as expected.
// RUN: FileCheck %s -input-file %t/result1.txt -DPREFIX=%/t -check-prefix ORDER

// ORDER:      {{.*}} - [[PREFIX]]/t.c
// ORDER-NEXT: [[PREFIX]]/t.c
// ORDER-NEXT: 1:1 <built-in>
// ORDER-NEXT:   [[PREFIX]]/top.h
// ORDER-NEXT:     [[PREFIX]]/n1.h
// ORDER-NEXT:       [[PREFIX]]/n2.h
// ORDER-NEXT:     [[PREFIX]]/n3.h
// ORDER-NEXT: [[PREFIX]]/n3.h
// ORDER-NEXT: [[PREFIX]]/n2.h
// ORDER-NEXT: Files:
// ORDER-NEXT: [[PREFIX]]/t.c
// ORDER-NEXT: [[PREFIX]]/top.h
// ORDER-NEXT: [[PREFIX]]/n1.h
// ORDER-NEXT: [[PREFIX]]/n2.h
// ORDER-NEXT: [[PREFIX]]/n3.h
// ORDER-NOT: [[PREFIX]]

// Full dependency output
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree-full -cas-path %t/cas > %t/deps.json

// RUN: cat %t/result1.txt > %t/full.txt
// RUN: echo "FULL DEPS START" >> %t/full.txt
// RUN: cat %t/deps.json | sed 's:\\\\\?:/:g' >> %t/full.txt

// RUN: FileCheck %s -DPREFIX=%/t -check-prefix=FULL -input-file %t/full.txt

// Capture the tree id from experimental-include-tree ; ensure that it matches
// the result from experimental-full.
// FULL: [[TREE_ID:llvmcas://[[:xdigit:]]+]] - [[PREFIX]]/t.c
// FULL: FULL DEPS START

// FULL-NEXT: {
// FULL-NEXT:   "modules": []
// FULL-NEXT:   "translation-units": [
// FULL-NEXT:     {
// FULL-NEXT:       "commands": [
// FULL-NEXT:         {
// FULL-NEXT:           "cache-key": "[[TU_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// FULL:                "clang-module-deps": []
// FULL:                "command-line": [
// FULL-NEXT:             "-cc1"
// FULL:                  "-fcas-path"
// FULL-NEXT:             "[[PREFIX]]/cas"
// FULL:                  "-disable-free"
// FULL:                  "-fcas-include-tree"
// FULL-NEXT:             "[[TREE_ID]]"
// FULL:                  "-fcache-compile-job"
// FULL:                  "-fsyntax-only"
// FULL:                  "-x"
// FULL-NEXT:             "c"
// FULL-NOT:              "t.c"
// FULL:                  "-main-file-name"
// FULL-NEXT:             "t.c"
// FULL-NOT:              "t.c"
// FULL:                ]
// FULL:                "executable": "clang"
// FULL:                "file-deps": [
// FULL-NEXT:             "[[PREFIX]]/t.c"
// FULL-NEXT:             "[[PREFIX]]/top.h"
// FULL-NEXT:             "[[PREFIX]]/n1.h"
// FULL-NEXT:             "[[PREFIX]]/n2.h"
// FULL-NEXT:             "[[PREFIX]]/n3.h"
// FULL-NEXT:             "[[PREFIX]]/n3.h"
// FULL-NEXT:             "[[PREFIX]]/n2.h"
// FULL-NEXT:           ]
// FULL:                "input-file": "[[PREFIX]]/t.c"
// FULL:              }
// FULL:            ]
// FULL:          }
// FULL:        ]
// FULL:      }

// Build the include-tree command
// RUN: %deps-to-rsp %t/deps.json --tu-index 0 > %t/tu.rsp
// RUN: %clang @%t/tu.rsp -Rcompile-job-cache 2> %t/t.err

// Check cache key.
// RUN: cp %t/full.txt %t/combined.txt
// RUN: cat %t/t.err >> %t/combined.txt
// RUN: FileCheck %s -input-file=%t/combined.txt -check-prefix=COMBINED

// COMBINED:        "commands": [
// COMBINED-NEXT:     {
// COMBINED-NEXT:       "cache-key": "[[TU_CACHE_KEY:llvmcas://[[:xdigit:]]+]]"
// COMBINED:      remark: compile job cache miss for '[[TU_CACHE_KEY]]'

//--- cdb.json.template
[
  {
    "directory": "DIR",
    "command": "clang -fsyntax-only DIR/t.c",
    "file": "DIR/t.c"
  }
]

//--- t.c

#include "top.h" // this is top
// CHECK: [[@LINE]]:1 [[PREFIX]]/top.h

// Skipped because of macro guard.
#include "n1.h"

#include "n3.h"
// CHECK-DAG: [[@LINE]]:1 [[PREFIX]]/n3.h

#include "n2.h"
// CHECK-DAG: [[@LINE]]:1 [[PREFIX]]/n2.h

//--- top.h
#ifndef _TOP_H_
#define _TOP_H_

#if WHATEVER
typedef int MyT;
#endif

#define WHATEVER 1
#include "n1.h"
// CHECK-DAG:   [[@LINE]]:1 [[PREFIX]]/n1.h

#include "n3.h"
// CHECK-DAG:   [[@LINE]]:1 [[PREFIX]]/n3.h

#define ANOTHER 2

struct S {
  int x;
};

#endif

//--- n1.h
#pragma once

int x1;
#include "n2.h"
// CHECK-DAG:     [[@LINE]]:1 [[PREFIX]]/n2.h

int x2;

//--- n2.h
void foo(void);

//--- n3.h
#ifndef _N3_H_
#define _N3_H_

int x3;

#endif

// More stuff after following '#endif', invalidate the macro guard optimization.
#define THIS_INVALIDATES_THE_MACRO_GUARD 1
