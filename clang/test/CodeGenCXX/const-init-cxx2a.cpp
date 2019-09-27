// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin -emit-llvm -o - %s -std=c++2a | FileCheck %s
// expected-no-diagnostics

// CHECK: @a = global i32 123,
int a = (delete new int, 123);
