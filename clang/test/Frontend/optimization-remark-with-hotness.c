// Generate instrumentation and sampling profile data.
// RUN: llvm-profdata merge \
// RUN:     %S/Inputs/optimization-remark-with-hotness.proftext \
// RUN:     -o %t.profdata
// RUN: llvm-profdata merge -sample \
// RUN:     %S/Inputs/optimization-remark-with-hotness-sample.proftext \
// RUN:     -o %t-sample.profdata
//
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name \
// RUN:     optimization-remark-with-hotness.c %s -emit-llvm-only \
// RUN:     -fprofile-instrument-use-path=%t.profdata -Rpass=inline \
// RUN:     -Rpass-analysis=inline -Rpass-missed=inline \
// RUN:     -fdiagnostics-show-hotness -verify
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name \
// RUN:     optimization-remark-with-hotness.c %s -emit-llvm-only \
// RUN:     -fprofile-sample-use=%t-sample.profdata -Rpass=inline \
// RUN:     -Rpass-analysis=inline -Rpass-missed=inline \
// RUN:     -fdiagnostics-show-hotness -fdiagnostics-hotness-threshold=10 \
// RUN:     -verify
// The clang version of the previous test.
// RUN: %clang -target x86_64-apple-macosx10.9 %s -c -emit-llvm -o /dev/null \
// RUN:     -fprofile-instr-use=%t.profdata -Rpass=inline \
// RUN:     -Rpass-analysis=inline -Rpass-missed=inline \
// RUN:     -fdiagnostics-show-hotness -fdiagnostics-hotness-threshold=10 \
// RUN:     -Xclang -verify
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name \
// RUN:     optimization-remark-with-hotness.c %s -emit-llvm-only \
// RUN:     -fprofile-instrument-use-path=%t.profdata -Rpass=inline \
// RUN:     -Rpass-analysis=inline 2>&1 | FileCheck -check-prefix=HOTNESS_OFF %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name \
// RUN:     optimization-remark-with-hotness.c %s -emit-llvm-only \
// RUN:     -fprofile-instrument-use-path=%t.profdata -Rpass=inline \
// RUN:     -Rpass-analysis=inline -Rno-pass-with-hotness 2>&1 | FileCheck \
// RUN:     -check-prefix=HOTNESS_OFF %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name \
// RUN:     optimization-remark-with-hotness.c %s -emit-llvm-only \
// RUN:     -fprofile-instrument-use-path=%t.profdata -Rpass=inline \
// RUN:     -Rpass-analysis=inline -fdiagnostics-show-hotness \
// RUN:     -fdiagnostics-hotness-threshold=100 2>&1 \
// RUN:     | FileCheck -allow-empty -check-prefix=THRESHOLD %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name \
// RUN:     optimization-remark-with-hotness.c %s -emit-llvm-only \
// RUN:     -Rpass=inline -Rpass-analysis=inline \
// RUN:     -fdiagnostics-show-hotness -fdiagnostics-hotness-threshold=10 2>&1 \
// RUN:     | FileCheck -check-prefix=NO_PGO %s

int foo(int x, int y) __attribute__((always_inline));
int foo(int x, int y) { return x + y; }

int sum = 0;

void bar(int x) {
  // HOTNESS_OFF: foo inlined into bar
  // HOTNESS_OFF-NOT: hotness:
  // THRESHOLD-NOT: inlined
  // THRESHOLD-NOT: hotness
  // NO_PGO: '-fdiagnostics-show-hotness' requires profile-guided optimization information
  // NO_PGO: '-fdiagnostics-hotness-threshold=' requires profile-guided optimization information
  // expected-remark@+1 {{foo inlined into bar with cost=always}}
  sum += foo(x, x - 2);
}

int main(int argc, const char *argv[]) {
  for (int i = 0; i < 30; i++)
    // expected-remark@+1 {{bar not inlined into main because it should never be inlined}}
    bar(argc);
  return sum;
}
