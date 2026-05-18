// REQUIRES: linux || darwin
// Default
// RUN: %clang -o %t.normal -fprofile-instr-generate -fcoverage-mapping -fcoverage-mcdc %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw
// RUN: llvm-profdata show --all-functions --counts --text %t.normal.profdata > %t.normal.profdata.show

// With -profile-correlate=binary flag
// RUN: %clang -o %t-1.exe -fprofile-instr-generate -fcoverage-mapping -fcoverage-mcdc -mllvm -profile-correlate=binary %s
// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t-1.exe
// RUN: llvm-profdata merge -o %t-1.profdata --binary-file=%t-1.exe %t-1.profraw
// RUN: llvm-profdata show --all-functions --counts --text %t-1.profdata > %t-1.profdata.show

// With -profile-correlate=debug-info flag
// RUN: %clang -o %t-2.exe -fprofile-instr-generate -fcoverage-mapping -fcoverage-mcdc -mllvm -profile-correlate=debug-info -g %s
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t-2.exe
// RUN: llvm-profdata merge -o %t-2.profdata --debug-info=%t-2.exe %t-2.profraw
// RUN: llvm-profdata show --all-functions --counts --text %t-2.profdata > %t-2.profdata.show

// RUN: diff %t.normal.profdata.show %t-1.profdata.show
// RUN: diff %t.normal.profdata.show %t-2.profdata.show
void test(int a, int b, int c, int d) {
  if ((a && b) || (c && d))
    ;
  if (b && c)
    ;
}

int main() {
  test(1, 1, 1, 1);
  test(1, 1, 0, 1);
  test(0, 0, 1, 0);
  test(0, 0, 1, 1);
  return 0;
}
