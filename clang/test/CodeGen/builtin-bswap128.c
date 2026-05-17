// RUN: %clang_cc1 -triple x86_64-linux-gnu -no-enable-noundef-analysis -emit-llvm -o - %s | FileCheck %s

#ifdef __SIZEOF_INT128__
__uint128_t test_constexpr(void) {
  return __builtin_bswap128(0x1234);
}

// CHECK-LABEL: define{{.*}} i128 @test_constexpr()
// CHECK: ret i128 69213317124269252288311516068503879680

__uint128_t test_non_const(__uint128_t x) {
  return __builtin_bswap128(x);
}

// CHECK-LABEL: define{{.*}} i128 @test_non_const(i128 %x)
// CHECK: call i128 @llvm.bswap.i128(i128 %{{.*}})
// CHECK: ret i128
#endif
