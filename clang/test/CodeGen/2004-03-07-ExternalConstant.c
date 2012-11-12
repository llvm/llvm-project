// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s

// CHECK: @a = external constan
extern const int a[];   // 'a' should be marked constant even though it's external!
int foo () {
  return a[0];
}
