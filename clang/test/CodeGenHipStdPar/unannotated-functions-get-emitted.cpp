// RUN: %clang_cc1 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck --check-prefix=NO-HIPSTDPAR-DEV %s

// RUN: %clang_cc1 --hipstdpar -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck --check-prefix=HIPSTDPAR-DEV %s

#define __device__ __attribute__((device))

// NO-HIPSTDPAR-DEV-NOT: define {{.*}} void @foo({{.*}})
// HIPSTDPAR-DEV: define {{.*}} void @foo({{.*}})
extern "C" void foo(float *a, float b) {
  *a = b;
}

// NO-HIPSTDPAR-DEV: define {{.*}} void @bar({{.*}})
// HIPSTDPAR-DEV: define {{.*}} void @bar({{.*}})
extern "C" __device__ void bar(float *a, float b) {
  *a = b;
}
