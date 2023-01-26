// AIX library functions frexpl, ldexpl, and modfl are for 128-bit IBM
// 'long double' (i.e. __ibm128). Check that the compiler generates
// calls to the 'double' versions for corresponding builtin functions in
// 64-bit 'long double' mode.

// RUN: %clang_cc1 -triple powerpc-ibm-aix -mlong-double-64 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -mlong-double-64 -emit-llvm -o - %s | FileCheck -check-prefix=CHECK %s

int main()
{
  int DummyInt;
  long double DummyLongDouble;
  long double returnValue;

  returnValue = __builtin_modfl(1.0L, &DummyLongDouble);
  returnValue = __builtin_frexpl(0.0L, &DummyInt);
  returnValue = __builtin_ldexpl(1.0L, 1);
}

// CHECK: %call = call double @modf(double noundef 1.000000e+00, ptr noundef %DummyLongDouble) #3
// CHECK: %call1 = call double @frexp(double noundef 0.000000e+00, ptr noundef %DummyInt) #3
// CHECK: %call2 = call double @ldexp(double noundef 1.000000e+00, i32 noundef {{(signext )?}}1) #4
