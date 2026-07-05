// REQUIRES: x86-registered-target
// Test with pch.
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-pch -o %t.pch %S/external-defs.h
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -include-pch %t.pch -emit-llvm -o %t %s

// RUN: grep "@x =.* global i32 0" %t | count 1
// RUN: not grep "@z" %t

// RUN: grep "@x2 =.* global i32 19" %t | count 1
int x2 = 19;

// RUN: grep "@incomplete_array =.* global .*1 x i32" %t | count 1
// RUN: grep "@incomplete_array2 =.* global .*17 x i32" %t | count 1
int incomplete_array2[17];
// RUN: grep "@incomplete_array3 =.* global .*1 x i32" %t | count 1
int incomplete_array3[];

struct S {
  int x, y;
};
