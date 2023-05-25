// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int average(int count, ...);
// CHECK: cir.func private @{{.*}}average{{.*}}(!s32i, ...) -> !s32i

int test(void) {
  return average(5, 1, 2, 3, 4, 5);
  // CHECK: cir.call @{{.*}}average{{.*}}(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}) : (!s32i, !s32i, !s32i, !s32i, !s32i, !s32i) -> !s32i
}
