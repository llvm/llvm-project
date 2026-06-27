// AIX library functions frexpl, ldexpl, and modfl are for 128-bit IBM
// 'long double' (i.e. __ibm128). Check that the compiler generates
// calls to the 'double' versions for corresponding builtin functions in
// 64-bit 'long double' mode.

// RUN: %clang_cc1 -triple powerpc-ibm-aix -mlong-double-64 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -mlong-double-64 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK %s

long double arg = 1.0L;
int main()
{
  int DummyInt;
  long double DummyLongDouble;
  long double returnValue;

  returnValue = __builtin_modfl(arg, &DummyLongDouble);
  returnValue = __builtin_frexpl(arg, &DummyInt);
  returnValue = __builtin_ldexpl(arg, 1);
}

// CHECK: %{{.+}} = call { double, double } @llvm.modf.f64(double %{{.+}})
// CHECK: %{{.+}} = call { double, i32 } @llvm.frexp.f64.i32(double %{{.+}})
// CHECK: %{{.+}} = call double @llvm.ldexp.f64.i32(double %{{.+}}, i32 1)
