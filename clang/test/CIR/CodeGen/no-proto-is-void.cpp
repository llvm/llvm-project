// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -std=c2x -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Both CXX and C2X don't support no-prototype functions. They default to void.
int noProto();
// CHECK: cir.func @{{.*}}noProto{{.*}}() -> !s32i
int test(int x) {
  return noProto();
  // CHECK {{.+}} = cir.call @{{.*}}noProto{{.*}}() : () -> !s32i
}
int noProto() { return 0; }
