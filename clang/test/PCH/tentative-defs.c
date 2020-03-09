// Test with pch.
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -emit-pch -o %t.pch %S/tentative-defs.h
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -include-pch %t.pch -verify -emit-llvm -o %t %s
// REQUIRES: x86-registered-target

// RUN: grep "@variable = common global i32 0" %t | count 1
// RUN: grep "@incomplete_array = common global .*1 x i32" %t | count 1

// FIXME: expected-warning@tentative-defs.h:9{{tentative}}
