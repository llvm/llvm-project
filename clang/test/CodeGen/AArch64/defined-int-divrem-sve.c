// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +sve -fdefined-int-divrem -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg | FileCheck %s

// -fdefined-int-divrem on scalable vectors: a splat constant divisor that is
// safe in every lane keeps the plain instruction.

// CHECK-LABEL: @sdiv_splat(
// CHECK:       [[R:%.*]] = sdiv <vscale x 4 x i32> %a, splat (i32 4)
// CHECK-NEXT:  ret <vscale x 4 x i32> [[R]]
__SVInt32_t sdiv_splat(__SVInt32_t a) { return a / 4; }

// CHECK-LABEL: @srem_minus_one(
// CHECK:       [[NMIN:%.*]] = icmp ne <vscale x 4 x i32> %a, splat (i32 -2147483648)
// CHECK-NEXT:  [[Q:%.*]] = call <vscale x 4 x i32> @llvm.masked.srem.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> splat (i32 -1), <vscale x 4 x i1> [[NMIN]])
// CHECK-NEXT:  freeze <vscale x 4 x i32> [[Q]]
__SVInt32_t srem_minus_one(__SVInt32_t a) { return a % -1; }

// CHECK-LABEL: @udiv_var(
// CHECK:       [[NZ:%.*]] = icmp ne <vscale x 4 x i32> %b, zeroinitializer
// CHECK-NEXT:  [[Q:%.*]] = call <vscale x 4 x i32> @llvm.masked.udiv.nxv4i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i1> [[NZ]])
// CHECK-NEXT:  freeze <vscale x 4 x i32> [[Q]]
__SVUint32_t udiv_var(__SVUint32_t a, __SVUint32_t b) { return a / b; }

static int obj;

// CHECK-LABEL: @sdiv_ptr(
// CHECK:       [[Q:%.*]] = call <vscale x 2 x i64> @llvm.masked.sdiv.nxv2i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> {{.*}}, <vscale x 2 x i1> {{.*}})
// CHECK-NEXT:  freeze <vscale x 2 x i64> [[Q]]
__SVInt64_t sdiv_ptr(__SVInt64_t a) { return a / (long)&obj; }
