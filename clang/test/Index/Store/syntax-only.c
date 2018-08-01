// RUN: rm -rf %t.idx
// RUN: %clang -fsyntax-only %s -index-store-path %t.idx -o %T/syntax-only.c.myoutfile
// RUN: c-index-test core -print-unit %t.idx | FileCheck %s -check-prefix=CHECK-UNIT
// RUN: c-index-test core -print-record %t.idx | FileCheck %s -check-prefix=CHECK-RECORD

// XFAIL: linux

// CHECK-UNIT: out-file: {{.*}}/syntax-only.c.myoutfile
// CHECK-RECORD: function/C | foo | c:@F@foo

void foo();
