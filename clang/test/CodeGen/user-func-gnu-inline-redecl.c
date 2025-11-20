// RUN: %clang_cc1 -triple x86_64 -emit-llvm -O1 -o - %s | FileCheck %s
//
// Verifies that the gnu_inline version is ignored in favor of the redecl

extern inline __attribute__((gnu_inline)) unsigned long some_size(int c) {
  return 1;
}
unsigned long mycall(int s) {
  // CHECK-LABEL: i64 @mycall
  // CHECK: ret i64 2
  return some_size(s);
}
unsigned long some_size(int c) {
  return 2;
}
unsigned long yourcall(int s) {
  // CHECK-LABEL: i64 @yourcall
  // CHECK: ret i64 2
  return some_size(s);
}
