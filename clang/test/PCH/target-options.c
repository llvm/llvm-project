// RUN: %clang_cc1 -triple=x86_64-apple-darwin9 -emit-pch -o %t.pch %S/target-options.h
// RUN: not %clang_cc1 -triple=x86_64-unknown-freebsd7.0 -include-pch %t.pch %s -emit-llvm -o - > %t.err 2>&1
// RUN: FileCheck %s < %t.err
// REQUIRES: x86-registered-target

// CHECK: for the target
