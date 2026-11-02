// RUN: %clang_cc1 -verify -triple x86_64-pc-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

extern "C" void body(int);

// CHECK-LABEL: define {{.*}}void @_Z11braced_initv(
// CHECK:         store i32 3, ptr %[[I:.+]], align 4
// CHECK:         icmp sgt i32 %{{.+}}, 1
// CHECK:         icmp slt i32 %{{.+}}, 4
// CHECK:         %[[MUL:.+]] = mul nsw i32 %{{.+}}, 1
// CHECK-NEXT:    %[[ADD:.+]] = add nsw i32 3, %[[MUL]]
// CHECK-NEXT:    store i32 %[[ADD]], ptr %[[I]], align 4
void braced_init() {
#pragma omp for
#pragma omp unroll partial(2)
  for (int i = {3}; i < 7; i++)
    body(i);
}

// CHECK-LABEL: define {{.*}}void @_Z10empty_initv(
// CHECK:         store i32 0, ptr %[[I:.+]], align 4
// CHECK:         icmp sgt i32 %{{.+}}, 1
// CHECK:         icmp slt i32 %{{.+}}, 4
// CHECK:         %[[MUL:.+]] = mul nsw i32 %{{.+}}, 1
// CHECK-NEXT:    %[[ADD:.+]] = add nsw i32 0, %[[MUL]]
// CHECK-NEXT:    store i32 %[[ADD]], ptr %[[I]], align 4
void empty_init() {
#pragma omp for
#pragma omp unroll partial(2)
  for (int i = {}; i < 4; i++)
    body(i);
}
