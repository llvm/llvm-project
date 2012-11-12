// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s

struct X { int V[10000]; };
struct X Global1, Global2;
void test() {
  // CHECK: llvm.memcpy
  Global2 = Global1;
}
