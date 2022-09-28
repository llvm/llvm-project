// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed -e "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: %clang -x c-header %t/prefix.h -target x86_64-apple-macos12 -o %t/prefix.pch -fdepscan=inline -fdepscan-include-tree -Xclang -fcas-path -Xclang %t/cas
// RUN: clang-scan-deps -compilation-database %t/cdb.json -format experimental-include-tree -cas-path %t/cas > %t/result.txt
// RUN: FileCheck %s -input-file %t/result.txt -DPREFIX=%/t

// CHECK:      {{.*}} - [[PREFIX]]/t.c
// CHECK-NEXT: (PCH)
// CHECK-NEXT: [[PREFIX]]/t.c
// CHECK-NEXT: 1:1 <built-in>
// CHECK-NEXT: [[PREFIX]]/t.h
// CHECK-NEXT: Files:
// CHECK-NEXT: [[PREFIX]]/t.c
// CHECK-NEXT: [[PREFIX]]/t.h
// CHECK-NEXT: [[PREFIX]]/prefix.h
// CHECK-NEXT: [[PREFIX]]/n1.h
// CHECK-NEXT: [[PREFIX]]/n2.h
// CHECK-NEXT: [[PREFIX]]/n3.h
// CHECK-NOT: [[PREFIX]]

//--- prefix.h
#include "n1.h"
#import "n2.h"
#include "n3.h"

//--- n1.h
#ifndef _N1_H_
#define _N1_H_

int n1 = 0;

#endif

//--- n2.h
int n2 = 0;

//--- n3.h
#pragma once
int n3 = 0;

//--- cdb.json.template
[{
  "directory" : "DIR",
  "command" : "clang -fsyntax-only DIR/t.c -target x86_64-apple-macos12 -isysroot DIR -Xclang -include-pch -Xclang DIR/prefix.pch",
  "file" : "DIR/t.c"
}]

//--- t.c
#include "t.h"
// Should not include due to macro guard.
#include "n1.h"
// Should not include due to #import from 'prefix.h'.
#include "n2.h"
// Should not include due to '#pragma once'
#include "n3.h"

//--- t.h
