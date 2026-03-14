// Test with fast math
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -DFAST \
// RUN: -mreassociate \
// RUN: -o - %s | FileCheck --check-prefixes CHECK,CHECKFAST,CHECKNP %s
//
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -DFAST \
// RUN: -mreassociate \
// RUN: -o - %s | FileCheck --check-prefixes CHECK,CHECKFAST,CHECKNP %s
//
// Test with fast math and fprotect-parens
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -DFAST \
// RUN: -mreassociate -fprotect-parens -ffp-contract=on\
// RUN: -o - %s | FileCheck --check-prefixes CHECK,CHECKFAST,CHECKPP %s
//
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -DFAST \
// RUN: -mreassociate -fprotect-parens -ffp-contract=on\
// RUN: -o - %s | FileCheck --check-prefixes CHECK,CHECKFAST,CHECKPP %s
//
// Test without fast math: llvm intrinsic not created
// RUN: %clang_cc1 -triple i386-pc-linux-gnu -emit-llvm -fprotect-parens\
// RUN: -o - %s | FileCheck --implicit-check-not="llvm.arithmetic.fence" %s
//
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fprotect-parens\
// RUN: -o - %s | FileCheck --implicit-check-not="llvm.arithmetic.fence" %s
//
// Test with fast math on spir target
// RUN: %clang_cc1 -triple spir64  -emit-llvm -DFAST \
// RUN: -mreassociate -o - %s \
// RUN: | FileCheck --check-prefixes CHECK,CHECKFAST,CHECKNP %s
//

int v;
int addit(float a, float b) {
  // CHECK: define {{.*}}@addit(float noundef %a, float noundef %b) #0 {
  _Complex double cd, cd1;
  cd = __arithmetic_fence(cd1);
  // CHECKFAST: call{{.*}} double @llvm.arithmetic.fence.f64({{.*}}real)
  // CHECKFAST: call{{.*}} double @llvm.arithmetic.fence.f64({{.*}}imag)
  // Vector should be supported.
  typedef float __v2f32 __attribute__((__vector_size__(8)));
  __v2f32 vec1, vec2;
  vec1 = __arithmetic_fence(vec2);
  // CHECKFAST: call{{.*}} <2 x float> @llvm.arithmetic.fence.v2f32
  vec2 = (vec2 + vec1);
  // CHECKPP: call{{.*}} <2 x float> @llvm.arithmetic.fence.v2f32

  v = __arithmetic_fence(a + b);
  // CHECKFAST: call{{.*}} float @llvm.arithmetic.fence.f32(float %add{{.*}})

  v = (a + b);
  // CHECKPP: call{{.*}} float @llvm.arithmetic.fence.f32(float %add{{.*}})
  v = a + (b*b);
  // CHECKPP: fmul reassoc
  // CHECKPP-NEXT: call{{.*}} float @llvm.arithmetic.fence.f32(float %mul)
  // CHECKNP: fmul
  // CHECKNP: fadd
  v = b + a*a;
  // CHECKPP: call{{.*}} float @llvm.fmuladd.f32
  // CHECKNP: fmul
  // CHECKNP: fadd
  v = b + __arithmetic_fence(a*a); // Fence blocks recognition of FMA
  // CHECKPP: fmul
  // CHECKNP: fmul

  b = (a);
  (a) = b;
  // CHECK-NEXT fptosi
  // CHECK-NEXT store i32
  // CHECK-NEXT load float
  // CHECK-NEXT store float
  // CHECK-NEXT load float
  // CHECK-NEXT store float
  return 0;
  // CHECK-NEXT ret i32 0
}
int addit1(int a, int b) {
  // CHECK: define {{.*}}@addit1(i32 noundef %a, i32 noundef %b{{.*}}
  v = (a + b);
  // CHECK-NOT: call{{.*}} float @llvm.arithmetic.fence.int(float noundef %add)
  return 0;
}
#ifdef FAST
#pragma float_control(precise, on)
int subit(float a, float b, float *fp) {
  // CHECKFAST: define {{.*}}@subit(float noundef %a, float noundef %b{{.*}}
  *fp = __arithmetic_fence(a - b);
  *fp = (a + b);
  // CHECK-NOT: call{{.*}} float @llvm.arithmetic.fence.f32(float noundef %add)
  return 0;
}
#endif
