// This test checks that -Wshadow-header doesn't repeatedly perform the same IO.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- tu1.c
#include "header.h"
//--- tu2.c
#include "header.h"
// The following line should not trigger more IO:
#include "header.h"
//--- include1/header.h
//--- include2/keep.h

// RUN: %clang_cc1 -Eonly %t/tu1.c -I %t/include1 -I %t/include2 -Wshadow-header -print-stats 2>%t/tu1.stats
// RUN: %clang_cc1 -Eonly %t/tu2.c -I %t/include1 -I %t/include2 -Wshadow-header -print-stats 2>%t/tu2.stats

// RUN: cat %t/tu1.stats %t/tu2.stats | FileCheck %s
// CHECK:      *** Virtual File System Stats:
// CHECK-NEXT: [[STATUS_COUNT:[0-9]+]] status() calls
// CHECK:      *** Virtual File System Stats:
// CHECK-NEXT: [[STATUS_COUNT]] status() calls
