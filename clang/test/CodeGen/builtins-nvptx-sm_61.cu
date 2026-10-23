// The builtins must work on sm_61 and newer (we also test on sm_90). The two
// runs below only look at the CHECK lines.
//
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -target-cpu sm_61 \
// RUN:            -fcuda-is-device -target-feature +ptx62 \
// RUN:            -emit-llvm -o - -x cuda %s | FileCheck %s
// RUN: %clang_cc1 -triple nvptx64-unknown-unknown -target-cpu sm_90 \
// RUN:            -fcuda-is-device -target-feature +ptx80 \
// RUN:            -emit-llvm -o - -x cuda %s | FileCheck %s
//
//
// The builtins must be rejected on older GPUs, tested on sm_60 with -verify.
// The CHECK lines are ignored by this run.
//
// RUN: %clang_cc1 -triple nvptx-unknown-unknown -target-cpu sm_60 \
// RUN:            -fcuda-is-device -target-feature +ptx62 \
// RUN:            -S -o /dev/null -x cuda -verify %s

#define __device__ __attribute__((device))

// CHECK-LABEL: @_Z11dot_productPiiiijjj(
__device__ void dot_product(int *out, int a, int b, int c, unsigned int ua,
                            unsigned int ub, unsigned int uc) {
  // CHECK: call i32 @llvm.nvvm.idp4a.s.s(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp4a_s_s' needs target feature sm_61}}
  out[0] = __nvvm_idp4a_s_s(a, b, c);
  // CHECK: call i32 @llvm.nvvm.idp4a.s.u(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp4a_s_u' needs target feature sm_61}}
  out[1] = __nvvm_idp4a_s_u(a, ub, c);
  // CHECK: call i32 @llvm.nvvm.idp4a.u.s(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp4a_u_s' needs target feature sm_61}}
  out[2] = __nvvm_idp4a_u_s(ua, b, c);
  // CHECK: call i32 @llvm.nvvm.idp4a.u.u(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp4a_u_u' needs target feature sm_61}}
  out[3] = __nvvm_idp4a_u_u(ua, ub, uc);

  // CHECK: call i32 @llvm.nvvm.idp2a.s.s(i32 %{{.*}}, i32 %{{.*}}, i1 false, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_lo_s_s' needs target feature sm_61}}
  out[4] = __nvvm_idp2a_lo_s_s(a, b, c);
  // CHECK: call i32 @llvm.nvvm.idp2a.s.u(i32 %{{.*}}, i32 %{{.*}}, i1 false, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_lo_s_u' needs target feature sm_61}}
  out[5] = __nvvm_idp2a_lo_s_u(a, ub, c);
  // CHECK: call i32 @llvm.nvvm.idp2a.u.s(i32 %{{.*}}, i32 %{{.*}}, i1 false, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_lo_u_s' needs target feature sm_61}}
  out[6] = __nvvm_idp2a_lo_u_s(ua, b, c);
  // CHECK: call i32 @llvm.nvvm.idp2a.u.u(i32 %{{.*}}, i32 %{{.*}}, i1 false, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_lo_u_u' needs target feature sm_61}}
  out[7] = __nvvm_idp2a_lo_u_u(ua, ub, uc);

  // CHECK: call i32 @llvm.nvvm.idp2a.s.s(i32 %{{.*}}, i32 %{{.*}}, i1 true, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_hi_s_s' needs target feature sm_61}}
  out[8] = __nvvm_idp2a_hi_s_s(a, b, c);
  // CHECK: call i32 @llvm.nvvm.idp2a.s.u(i32 %{{.*}}, i32 %{{.*}}, i1 true, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_hi_s_u' needs target feature sm_61}}
  out[9] = __nvvm_idp2a_hi_s_u(a, ub, c);
  // CHECK: call i32 @llvm.nvvm.idp2a.u.s(i32 %{{.*}}, i32 %{{.*}}, i1 true, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_hi_u_s' needs target feature sm_61}}
  out[10] = __nvvm_idp2a_hi_u_s(ua, b, c);
  // CHECK: call i32 @llvm.nvvm.idp2a.u.u(i32 %{{.*}}, i32 %{{.*}}, i1 true, i32 %{{.*}})
  // expected-error@+1 {{'__nvvm_idp2a_hi_u_u' needs target feature sm_61}}
  out[11] = __nvvm_idp2a_hi_u_u(ua, ub, uc);
  // CHECK: ret void
}
