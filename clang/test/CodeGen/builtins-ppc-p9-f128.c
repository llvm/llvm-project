// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm \
// RUN:   -target-cpu pwr9 -target-feature +float128 -o - %s | FileCheck %s

__float128 A;
__float128 B;
__float128 C;


__float128 testSqrtOdd() {
  return __builtin_sqrtf128_round_to_odd(A);
// CHECK: @llvm.ppc.sqrtf128.round.to.odd(fp128
// CHECK-NEXT: ret fp128
}

__float128 testFMAOdd() {
  return __builtin_fmaf128_round_to_odd(A, B, C);
// CHECK: @llvm.ppc.fmaf128.round.to.odd(fp128 %{{.+}}, fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testAddOdd() {
  return __builtin_addf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.addf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testSubOdd() {
  return __builtin_subf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.subf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testMulOdd() {
  return __builtin_mulf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.mulf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

__float128 testDivOdd() {
  return __builtin_divf128_round_to_odd(A, B);
// CHECK: @llvm.ppc.divf128.round.to.odd(fp128 %{{.+}}, fp128
// CHECK-NEXT: ret fp128
}

double testTruncOdd() {
  return __builtin_truncf128_round_to_odd(A);
// CHECK: @llvm.ppc.truncf128.round.to.odd(fp128
// CHECK-NEXT: ret double
}

