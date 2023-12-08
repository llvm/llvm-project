// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu pwr7 \
// RUN: -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr8 \
// RUN: -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} signext i32 @test_divwe
int test_divwe(void)
{
  int a = 74;
  int b = 32;
  return __builtin_divwe(a, b);
// CHECK: @llvm.ppc.divwe
}

// CHECK-LABEL: define{{.*}} zeroext i32 @test_divweu
unsigned int test_divweu(void)
{
  unsigned int a = 74;
  unsigned int b = 32;
  return __builtin_divweu(a, b);
// CHECK: @llvm.ppc.divweu
}

// CHECK-LABEL: define{{.*}} i64 @test_divde
long long test_divde(void)
{
  long long a = 74LL;
  long long b = 32LL;
  return __builtin_divde(a, b);
// CHECK: @llvm.ppc.divde
}

// CHECK-LABEL: define{{.*}} i64 @test_divdeu
unsigned long long test_divdeu(void)
{
  unsigned long long a = 74ULL;
  unsigned long long b = 32ULL;
  return __builtin_divdeu(a, b);
// CHECK: @llvm.ppc.divdeu
}

// CHECK-LABEL: define{{.*}} i64 @test_bpermd
long long test_bpermd(void)
{
  long long a = 74LL;
  long long b = 32LL;
  return __builtin_bpermd(a, b);
// CHECK: @llvm.ppc.bpermd
}

