// REQUIRES: zlib

// Value profiling is currently not supported in lightweight mode.
// RUN: %clang_pgogen -o %t.normal -mllvm --disable-vp=true %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: %clang_pgogen -o %t -g -mllvm --debug-info-correlate -mllvm --disable-vp=true %s
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t %t.proflite

// RUN: diff %t.normal.profdata %t.profdata

int foo(int a) {
  if (a % 2)
    return 4 * a + 1;
  return 0;
}

int bar(int a) {
  while (a > 100)
    a /= 2;
  return a;
}

typedef int (*FP)(int);
FP Fps[3] = {foo, bar};

int main() {
  for (int i = 0; i < 5; i++)
    Fps[i % 2](i);
  return 0;
}
