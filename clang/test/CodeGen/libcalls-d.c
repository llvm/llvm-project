// llvm-gcc -O1+ should run simplify libcalls, O0 shouldn't
// and -fno-builtins shouldn't.
// -fno-math-errno should emit an llvm intrinsic, -fmath-errno should not.
// RUN: %clang_cc1 %s -emit-llvm -fno-math-errno -O0 -o - | grep {call.*exp2\\.f64}
// RUN: %clang_cc1 %s -emit-llvm -fmath-errno -O0 -o - | grep {call.*exp2}
// RUN: %clang_cc1 %s -emit-llvm -O1 -o - | grep {call.*ldexp}
// RUN: %clang_cc1 %s -emit-llvm -O3 -fno-builtin -o - | grep {call.*exp2}

// clang doesn't support this yet.
// XFAIL: *

double exp2(double);

double t4(unsigned char x) {
  return exp2(x);
}
