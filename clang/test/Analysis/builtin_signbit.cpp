// RUN: %clang -target powerpc-linux-gnu -emit-llvm -S -mabi=ibmlongdouble \
// RUN:   -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -target powerpc64-linux-gnu -emit-llvm -S -mabi=ibmlongdouble \
// RUN:   -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -target powerpc64le-linux-gnu -emit-llvm -S -mabi=ibmlongdouble \
// RUN:   -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LE

bool b;
double d = -1.0;
long double ld = -1.0L;
void test_signbit()
{
  b = __builtin_signbit(1.0L);
  // CHECK: i128
  // CHECK-LE-NOT: lshr
  // CHECK-BE: lshr
  // CHECK: bitcast
  // CHECK: ppc_fp128

  b = __builtin_signbit(ld);
  // CHECK: bitcast
  // CHECK: ppc_fp128
  // CHECK-LE-NOT: lshr
  // CHECK-BE: lshr

  b = __builtin_signbitf(1.0);
  // CHECK: store i8 0

  b = __builtin_signbitf(d);
  // CHECK: bitcast
  // CHECK-LE-NOT: lshr
  // CHECK-BE-NOT: lshr

  b = __builtin_signbitl(1.0L);
  // CHECK: i128
  // CHECK-LE-NOT: lshr
  // CHECK-BE: lshr
  // CHECK: bitcast
  // CHECK: ppc_fp128

  b = __builtin_signbitl(ld);
  // CHECK: bitcast
  // CHECK: ppc_fp128
  // CHECK-LE-NOT: lshr
  // CHECK-BE: lshr
}
