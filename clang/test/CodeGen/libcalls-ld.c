// llvm-gcc -O1+ should run simplify libcalls, O0 shouldn't
// and -fno-builtins shouldn't.
// -fno-math-errno should emit an llvm intrinsic, -fmath-errno should not.
// RUN: %clang_cc1 %s -emit-llvm -fno-math-errno -o - | grep {call.*exp2\\..*f}
// RUN: %clang_cc1 %s -emit-llvm -fmath-errno -o - | grep {call.*exp2l}
// RUN: %clang_cc1 %s -emit-llvm -O1 -o - | grep {call.*ldexp}
// RUN: %clang_cc1 %s -emit-llvm -O3 -fno-builtin -o - | grep {call.*exp2l}

// clang doesn't support this yet.
// XFAIL: *

// If this fails for you because your target doesn't support long double,
// please xfail the test.

long double exp2l(long double);

long double t4(unsigned char x) {
  return exp2l(x);
}
