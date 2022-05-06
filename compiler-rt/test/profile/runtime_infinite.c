// XFAIL: aix
// The waiting loop never exits via the normal
// path before the profile is dumped and the
// program is terminated. This tests checks
// that the entry of main is properly instrumented
// and has non-zero count.

// RUN: %clang_pgogen -mllvm -do-counter-promotion=false -O2 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata show -function main -counts  %t.profraw| FileCheck  %s
void exit(int);

int __llvm_profile_dump(void);
void __llvm_profile_reset_counters(void);

int g = 0;
__attribute__((noinline)) void doSth() {
  g++;

  if (g > 10000) {
    // dump profile and exit;
    __llvm_profile_dump();
    exit(0);
  }
}
int errorcode = 0;
int noerror() { return (errorcode == 0); }

int main(int argc, const char *argv[]) {
  //  waiting_loop
  while (noerror()) {
    doSth();
  }
}

// CHECK-LABEL: main
// CHECK: [10001, 1]
