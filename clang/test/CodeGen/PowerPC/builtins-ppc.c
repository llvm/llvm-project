// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-unknown -emit-llvm %s -o - | FileCheck %s

void test_eh_return_data_regno()
{
  volatile int res;
  res = __builtin_eh_return_data_regno(0);  // CHECK: store volatile i32 3
  res = __builtin_eh_return_data_regno(1);  // CHECK: store volatile i32 4
}

// CHECK-LABEL: define{{.*}} i64 @test_builtin_ppc_get_timebase
long long test_builtin_ppc_get_timebase() {
  // CHECK: call i64 @llvm.readcyclecounter()
  return __builtin_ppc_get_timebase();
}

void test_builtin_ppc_setrnd() {
  volatile double res;
  volatile int x = 100;
  
  // CHECK: call double @llvm.ppc.setrnd(i32 2)
  res = __builtin_setrnd(2);

  // CHECK: call double @llvm.ppc.setrnd(i32 100)
  res = __builtin_setrnd(100);

  // CHECK: call double @llvm.ppc.setrnd(i32 %2)
  res = __builtin_setrnd(x);
}

void test_builtin_ppc_flm() {
  volatile double res;
  // CHECK: call double @llvm.ppc.readflm()
  res = __builtin_readflm();

  // CHECK: call double @llvm.ppc.setflm(double %1)
  res = __builtin_setflm(res);
}

double test_builtin_unpack_ldbl(long double x) {
  // CHECK: call double @llvm.ppc.unpack.longdouble(ppc_fp128 %0, i32 1)
  return __builtin_unpack_longdouble(x, 1);
}

long double test_builtin_pack_ldbl(double x, double y) {
  // CHECK: call ppc_fp128 @llvm.ppc.pack.longdouble(double %0, double %1)
  return __builtin_pack_longdouble(x, y);
}

void test_builtin_ppc_maxminfe(long double a, long double b, long double c,
                               long double d) {
  volatile long double res;
  // CHECK: call ppc_fp128 (ppc_fp128, ppc_fp128, ppc_fp128, ...) @llvm.ppc.maxfe(ppc_fp128 %0, ppc_fp128 %1, ppc_fp128 %2, ppc_fp128 %3)
  res = __builtin_ppc_maxfe(a, b, c, d);

  // CHECK: call ppc_fp128 (ppc_fp128, ppc_fp128, ppc_fp128, ...) @llvm.ppc.minfe(ppc_fp128 %5, ppc_fp128 %6, ppc_fp128 %7, ppc_fp128 %8)
  res = __builtin_ppc_minfe(a, b, c, d);
}

void test_builtin_ppc_maxminfl(double a, double b, double c, double d) {
  volatile double res;
  // CHECK: call double (double, double, double, ...) @llvm.ppc.maxfl(double %0, double %1, double %2, double %3)
  res = __builtin_ppc_maxfl(a, b, c, d);

  // CHECK: call double (double, double, double, ...) @llvm.ppc.minfl(double %5, double %6, double %7, double %8)
  res = __builtin_ppc_minfl(a, b, c, d);
}

void test_builtin_ppc_maxminfs(float a, float b, float c, float d) {
  volatile float res;
  // CHECK: call float (float, float, float, ...) @llvm.ppc.maxfs(float %0, float %1, float %2, float %3)
  res = __builtin_ppc_maxfs(a, b, c, d);

  // CHECK: call float (float, float, float, ...) @llvm.ppc.minfs(float %5, float %6, float %7, float %8)
  res = __builtin_ppc_minfs(a, b, c, d);
}
