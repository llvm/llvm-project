// Integer division and remainder by zero and INT_MIN / -1 stay undefined in C
// by default; -fdefined-int-divrem gives them an unspecified result instead.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-matrix -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OFF --implicit-check-not=llvm.masked
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-matrix -fdefined-int-divrem -fno-defined-int-divrem -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,OFF --implicit-check-not=llvm.masked
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-matrix -fdefined-int-divrem -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ON
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-matrix -fdefined-int-divrem -x c++ -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ON,CXXON

// The UBSan checks are emitted as before, ahead of the operation.
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fenable-matrix -fdefined-int-divrem -fsanitize=integer-divide-by-zero,signed-integer-overflow-divide -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,UBSAN

#ifdef __cplusplus
extern "C" {
#endif

// CHECK-LABEL: define {{.*}} @sdiv(
// OFF:         sdiv i32
// ON:          [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 {{.*}}, i32 {{.*}}, i1 {{.*}})
// ON-NEXT:     freeze i32 [[Q]]
// UBSAN:       call void @__ubsan_handle_divrem_overflow_abort(
// UBSAN:       [[Q:%.*]] = call i32 @llvm.masked.sdiv.i32(i32 {{.*}}, i32 {{.*}}, i1 {{.*}})
// UBSAN-NEXT:  freeze i32 [[Q]]
int sdiv(int a, int b) { return a / b; }

// CHECK-LABEL: define {{.*}} @urem(
// OFF:         urem i32
// ON:          [[Q:%.*]] = call i32 @llvm.masked.urem.i32(i32 {{.*}}, i32 {{.*}}, i1 {{.*}})
// ON-NEXT:     freeze i32 [[Q]]
// UBSAN:       call void @__ubsan_handle_divrem_overflow_abort(
// UBSAN:       [[Q:%.*]] = call i32 @llvm.masked.urem.i32(i32 {{.*}}, i32 {{.*}}, i1 {{.*}})
unsigned urem(unsigned a, unsigned b) { return a % b; }

// CHECK-LABEL: define {{.*}} @div128(
// OFF:         sdiv i128
// ON:          [[Q:%.*]] = call i128 @llvm.masked.sdiv.i128(i128 {{.*}}, i128 {{.*}}, i1 {{.*}})
// ON-NEXT:     freeze i128 [[Q]]
_BitInt(128) div128(_BitInt(128) a, _BitInt(128) b) { return a / b; }

// CHECK-LABEL: define {{.*}} @compound(
// OFF:         srem i64
// ON:          [[Q:%.*]] = call i64 @llvm.masked.srem.i64(i64 {{.*}}, i64 {{.*}}, i1 {{.*}})
// ON-NEXT:     freeze i64 [[Q]]
long long compound(long long a, long long b) {
  a %= b;
  return a;
}

typedef int v4i __attribute__((vector_size(16)));
// CHECK-LABEL: define {{.*}} @vdiv(
// OFF:         sdiv <4 x i32>
// ON:          [[Q:%.*]] = call <4 x i32> @llvm.masked.sdiv.v4i32(<4 x i32> {{.*}}, <4 x i32> {{.*}}, <4 x i1> {{.*}})
// ON-NEXT:     freeze <4 x i32> [[Q]]
v4i vdiv(v4i a, v4i b) { return a / b; }

// The address of an object is a constant, but not one known to be safe.
static int obj;
// CHECK-LABEL: define {{.*}} @div_by_address(
// OFF:         sdiv i64 {{.*}}, ptrtoint (ptr @{{.*}}obj to i64)
// ON:          [[Q:%.*]] = call i64 @llvm.masked.sdiv.i64(i64 {{.*}}, i64 ptrtoint (ptr @{{.*}}obj to i64), i1 {{.*}})
// ON-NEXT:     freeze i64 [[Q]]
long div_by_address(long a) { return a / (long)&obj; }

// Matrix division by a scalar is not changed.
typedef int m2x2 __attribute__((matrix_type(2, 2)));
// CHECK-LABEL: define {{.*}} @matdiv(
// CHECK-NOT:   llvm.masked
// CHECK:       sdiv <4 x i32>
m2x2 matdiv(m2x2 m, int b) { return m / b; }

#ifdef __cplusplus
typedef long v2l __attribute__((vector_size(16)));

// CXXON-LABEL: define {{.*}} @vdiv_safe_const(
// CXXON-NOT:   llvm.masked
// CXXON:       sdiv <4 x i32> {{.*}}, <i32 1, i32 -2, i32 3, i32 -4>
v4i vdiv_safe_const(v4i a) { return a / v4i{1, -2, 3, -4}; }

// CXXON-LABEL: define {{.*}} @vdiv_mixed_zero(
// CXXON:       call <4 x i32> @llvm.masked.sdiv.v4i32(<4 x i32> {{.*}}, <4 x i32> <i32 2, i32 0, i32 4, i32 5>, <4 x i1>
v4i vdiv_mixed_zero(v4i a) { return a / v4i{2, 0, 4, 5}; }

// CXXON-LABEL: define {{.*}} @vdiv_ptr_lane(
// CXXON:       call <2 x i64> @llvm.masked.sdiv.v2i64(<2 x i64> {{.*}}, <2 x i64> <i64 ptrtoint (ptr @{{.*}}obj to i64), i64 2>, <2 x i1>
v2l vdiv_ptr_lane(v2l a) { return a / v2l{(long)&obj, 2}; }
}
#endif
