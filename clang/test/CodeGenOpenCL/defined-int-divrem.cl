// OpenCL C v3.0 s6.5.1: integer division by zero and INT_MIN / -1 result in
// an unspecified value, so by default integer / and % lower to a frozen
// llvm.<op>.poison intrinsic. A constant divisor that rules both cases out
// keeps the plain instruction.

// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -cl-std=CL3.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -cl-std=CL3.0 -fno-defined-int-divrem -emit-llvm -o - %s | FileCheck %s --check-prefix=OFF --implicit-check-not=.poison
// RUN: %clang_cc1 -disable-llvm-passes -triple spir64-unknown-unknown -cl-std=CL3.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -disable-llvm-passes -triple spirv64-unknown-unknown -cl-std=CL3.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -disable-llvm-passes -triple spirv64-unknown-unknown -cl-std=CL3.0 -fno-defined-int-divrem -emit-llvm -o - %s | FileCheck %s --check-prefix=OFF --implicit-check-not=.poison
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -cl-std=clc++2021 -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-amd-amdhsa -cl-std=CL3.0 -emit-llvm -o - %s | FileCheck %s --check-prefix=ON
// RUN: %clang_cc1 -disable-llvm-passes -triple x86_64-unknown-unknown -cl-std=CL3.0 -fsanitize=integer-divide-by-zero,signed-integer-overflow -emit-llvm -o - %s | FileCheck %s --check-prefix=UBSAN

typedef int int4 __attribute__((ext_vector_type(4)));
typedef unsigned long ulong;

int f_sdiv(int a, int b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_sdiv
// ON: = call i32 @llvm.sdiv.poison.i32(i32
// ON-NEXT: freeze i32
// The recoverable UBSan check comes first and the division stays defined.
// UBSAN-LABEL: define {{.*}} @{{.*}}f_sdiv
// UBSAN: call void @__ubsan_handle_divrem_overflow(
// UBSAN: = call i32 @llvm.sdiv.poison.i32(i32
// UBSAN-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_sdiv
// OFF: = sdiv i32

int f_srem(int a, int b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_srem
// ON: = call i32 @llvm.srem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_srem
// OFF: = srem i32

unsigned f_udiv(unsigned a, unsigned b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_udiv
// ON: = call i32 @llvm.udiv.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_udiv
// OFF: = udiv i32

unsigned f_urem(unsigned a, unsigned b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_urem
// ON: = call i32 @llvm.urem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_urem
// OFF: = urem i32

int4 f_vdiv(int4 a, int4 b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_vdiv
// ON: = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32>
// ON-NEXT: freeze <4 x i32>
// OFF-LABEL: define {{.*}} @{{.*}}f_vdiv
// OFF: = sdiv <4 x i32>

// A constant divisor other than 0 or -1 cannot overflow: plain sdiv.
int f_divc(int a) { return a / 4; }
// ON-LABEL: define {{.*}} @{{.*}}f_divc
// ON: = sdiv i32 {{.*}}, 4
// OFF-LABEL: define {{.*}} @{{.*}}f_divc
// OFF: = sdiv i32 {{.*}}, 4

// So does a splat of one in a vector division.
int4 f_vdivc(int4 a) { return a / 4; }
// ON-LABEL: define {{.*}} @{{.*}}f_vdivc
// ON: = sdiv <4 x i32> {{.*}}, splat (i32 4)
// OFF-LABEL: define {{.*}} @{{.*}}f_vdivc
// OFF: = sdiv <4 x i32> {{.*}}, splat (i32 4)

int4 f_vdivm1(int4 a) { return a / -1; }
// ON-LABEL: define {{.*}} @{{.*}}f_vdivm1
// ON: = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32> {{.*}}, <4 x i32> splat (i32 -1))
// ON-NEXT: freeze <4 x i32>
// OFF-LABEL: define {{.*}} @{{.*}}f_vdivm1
// OFF: = sdiv <4 x i32> {{.*}}, splat (i32 -1)

// A constant divisor of 0 is the case the intrinsic exists for.
int f_div0(int a) { return a / 0; }
// ON-LABEL: define {{.*}} @{{.*}}f_div0
// ON: = call i32 @llvm.sdiv.poison.i32(i32 {{.*}}, i32 0)
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_div0
// OFF: = sdiv i32 {{.*}}, 0

unsigned f_urem0(unsigned a) { return a % 0; }
// ON-LABEL: define {{.*}} @{{.*}}f_urem0
// ON: = call i32 @llvm.urem.poison.i32(i32 {{.*}}, i32 0)
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_urem0
// OFF: = urem i32 {{.*}}, 0

// A constant divisor of -1 may overflow (INT_MIN / -1): intrinsic.
int f_divm1(int a) { return a / -1; }
// ON-LABEL: define {{.*}} @{{.*}}f_divm1
// ON: = call i32 @llvm.sdiv.poison.i32(i32 {{.*}}, i32 -1)
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_divm1
// OFF: = sdiv i32 {{.*}}, -1

// Compound assignments go through the same path.
int f_divassign(int a, int b) {
  a /= b;
  return a;
}
// ON-LABEL: define {{.*}} @{{.*}}f_divassign
// ON: = call i32 @llvm.sdiv.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_divassign
// OFF: = sdiv i32

// Variable scalar divisors are splatted before the vector intrinsic call.
int4 f_vdiv_scalar(int4 a, int b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_vdiv_scalar
// ON: [[SPLAT:%.*]] = shufflevector <4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i32> zeroinitializer
// ON-NEXT: = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32> %{{.*}}, <4 x i32> [[SPLAT]])
// ON-NEXT: freeze <4 x i32>
// OFF-LABEL: define {{.*}} @{{.*}}f_vdiv_scalar
// OFF: = sdiv <4 x i32>

int4 f_vrem(int4 a, int4 b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_vrem
// ON: = call <4 x i32> @llvm.srem.poison.v4i32(<4 x i32>
// ON-NEXT: freeze <4 x i32>
// OFF-LABEL: define {{.*}} @{{.*}}f_vrem
// OFF: = srem <4 x i32>

long f_sdiv_long(long a, long b) { return a / b; }
// ON-LABEL: define {{.*}} @{{.*}}f_sdiv_long
// ON: = call i64 @llvm.sdiv.poison.i64(i64
// ON-NEXT: freeze i64
// OFF-LABEL: define {{.*}} @{{.*}}f_sdiv_long
// OFF: = sdiv i64

ulong f_urem_ulong(ulong a, ulong b) { return a % b; }
// ON-LABEL: define {{.*}} @{{.*}}f_urem_ulong
// ON: = call i64 @llvm.urem.poison.i64(i64
// ON-NEXT: freeze i64
// OFF-LABEL: define {{.*}} @{{.*}}f_urem_ulong
// OFF: = urem i64

int f_remassign(int a, int b) {
  a %= b;
  return a;
}
// ON-LABEL: define {{.*}} @{{.*}}f_remassign
// ON: = call i32 @llvm.srem.poison.i32(i32
// ON-NEXT: freeze i32
// OFF-LABEL: define {{.*}} @{{.*}}f_remassign
// OFF: = srem i32

// A multi-element vector literal is emitted through a compound-literal
// temporary, so the divisor is a load, not a constant. The intrinsic is
// used even when every lane is safe.
int4 f_vdiv_literal_splat(int4 a) { return a / (int4)(4, 4, 4, 4); }
// ON-LABEL: define {{.*}} @{{.*}}f_vdiv_literal_splat
// ON: = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32>
// ON-NEXT: freeze <4 x i32>
// OFF-LABEL: define {{.*}} @{{.*}}f_vdiv_literal_splat
// OFF: = sdiv <4 x i32>

int4 f_vdiv_literal_safe(int4 a) { return a / (int4)(1, 2, 3, 4); }
// ON-LABEL: define {{.*}} @{{.*}}f_vdiv_literal_safe
// ON: = call <4 x i32> @llvm.sdiv.poison.v4i32(<4 x i32>
// ON-NEXT: freeze <4 x i32>
// OFF-LABEL: define {{.*}} @{{.*}}f_vdiv_literal_safe
// OFF: = sdiv <4 x i32>
