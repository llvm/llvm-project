// RUN: %clang_profgen -O2 -mllvm -enable-value-profiling=true -mllvm -vp-static-alloc=true -mllvm -vp-counters-per-site=3 -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --all-functions -ic-targets  %t.profdata | FileCheck %s

// IR level instrumentation
// RUN: %clang_pgogen -O2 -mllvm -disable-vp=false -mllvm -vp-static-alloc=true  -mllvm -vp-counters-per-site=3 -o %t.ir  %s
// RUN: env LLVM_PROFILE_FILE=%t.ir.profraw %run %t.ir
// RUN: llvm-profdata merge -o %t.ir.profdata %t.ir.profraw
// RUN: llvm-profdata show --all-functions -ic-targets  %t.ir.profdata | FileCheck  %s

// IR level instrumentation, dynamic allocation
// RUN: %clang_pgogen -O2 -mllvm -disable-vp=false -mllvm -vp-static-alloc=false -o %t.ir.dyn  %s
// RUN: env LLVM_PROFILE_FILE=%t.ir.dyn.profraw %run %t.ir.dyn
// RUN: llvm-profdata merge -o %t.ir.dyn.profdata %t.ir.dyn.profraw
// RUN: llvm-profdata show --all-functions -ic-targets  %t.ir.dyn.profdata | FileCheck  %s
void callee_0() {}
void callee_1() {}
void callee_2() {}

void *CalleeAddrs[] = {callee_0, callee_1, callee_2, callee_2, callee_2};
extern void lprofSetMaxValsPerSite(unsigned);
extern void __llvm_profile_reset_counters();

typedef void (*FPT)(void);


// Testing value profiling eviction algorithm.
FPT getCalleeFunc(int I) { return CalleeAddrs[I]; }

int main() {
  int I;

  // First fill up two value profile entries with two targets
  lprofSetMaxValsPerSite(2);

  for (I = 0; I < 5; I++) {
    if (I == 2) {
      __llvm_profile_reset_counters();
    }
    // CHECK:  callee_2, 3
    // CHECK-NEXT: callee_1, 0
    // CHECK-NOT: callee_0,
    FPT FP = getCalleeFunc(I);
    FP();
  }
}
