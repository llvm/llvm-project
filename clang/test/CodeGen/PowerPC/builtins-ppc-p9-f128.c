// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -o - %s | FileCheck %s

__float128 A;
__float128 B;
__float128 C;


__float128 testSqrtOdd(void) {
  return __builtin_sqrtf128_round_to_odd(A);
// CHECK: @llvm.ppc.sqrtf128.round.to.odd(fp128
// CHECK-NEXT: ret fp128
}

__float128 testFMAOdd(void) {
  return __builtin_fmaf128_round_to_odd(A, B, C);
// CHECK: @llvm.ppc.fmaf128.round.to.odd(fp128 %{{.+}}, fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testAddOdd(void) {
  return __builtin_addf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.addf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testSubOdd(void) {
  return __builtin_subf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.subf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testMulOdd(void) {
  return __builtin_mulf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.mulf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testDivOdd(void) {
  return __builtin_divf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.divf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

double testTruncOdd(void) {
  return __builtin_truncf128_round_to_odd(A);
// CHECK: @llvm.ppc.truncf128.round.to.odd(fp128
// CHECK-NEXT: ret double
}

__float128 insert_exp_qp(unsigned long long int b) {
  return __builtin_vsx_scalar_insert_exp_qp(A, b);
// CHECK: @llvm.ppc.scalar.insert.exp.qp(fp128 %{{.+}}, i64
// CHECK-NEXT: ret fp128
}

unsigned long long int extract_exp(void) {
  return __builtin_vsx_scalar_extract_expq(A);
// CHECK: @llvm.ppc.scalar.extract.expq(fp128
// CHECK-NEXT: ret i64
}

int test_data_class_f128(__float128 q) {
  return __builtin_ppc_test_data_class(q, 0);
// CHECK-LABEL: @test_data_class_f128
// CHECK: [[TMP:%.*]] = call i32 @llvm.ppc.test.data.class.f128(fp128 %0, i32 0)
// CHECK-NEXT: ret i32 [[TMP]]
}
