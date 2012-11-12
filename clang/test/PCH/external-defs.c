// Test with pch.
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-pch -o %t.pch %S/external-defs.h
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -include-pch %t.pch -emit-llvm -o %t %s

// RUN: grep "@x = common global i32 0" %t | count 1
// RUN: grep "@z" %t | count 0

// RUN: grep "@x2 = global i32 19" %t | count 1
int x2 = 19;

// RUN: grep "@incomplete_array = common global .*1 x i32" %t | count 1
// RUN: grep "@incomplete_array2 = common global .*17 x i32" %t | count 1
int incomplete_array2[17];
// RUN: grep "@incomplete_array3 = common global .*1 x i32" %t | count 1
int incomplete_array3[];

struct S {
  int x, y;
};
