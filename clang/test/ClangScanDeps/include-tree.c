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
// ORDER-NOT: [[PREFIX]]


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
