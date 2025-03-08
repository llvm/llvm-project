// REQUIRES: continuous-mode

// RUN: %clang_profgen_cont -fcoverage-mapping -o %t.exe %s
// RUN: echo "garbage" > %t.profraw
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe
// RUN: llvm-profdata show --counts --all-functions %t.profraw | FileCheck %s -check-prefix=CHECK-COUNTS
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
//
// COM: The "report" and "show" commands of llvm-cov are not supported on AIX.
// RUN: %if !target={{.*aix.*}} %{ \
// RUN:   llvm-cov report %t.exe -instr-profile %t.profdata | FileCheck %s -check-prefix=CHECK-COVERAGE \
// RUN: %}

// CHECK-COUNTS: Counters:
// CHECK-COUNTS-NEXT:   main:
// CHECK-COUNTS-NEXT:     Hash: 0x{{.*}}
// CHECK-COUNTS-NEXT:     Counters: 2
// CHECK-COUNTS-NEXT:     Function count: 1
// CHECK-COUNTS-NEXT:     Block counts: [1]
// CHECK-COUNTS-NEXT: Instrumentation level: Front-end
// CHECK-COUNTS-NEXT: Functions shown: 1
// CHECK-COUNTS-NEXT: Total functions: 1
// CHECK-COUNTS-NEXT: Maximum function count: 1
// CHECK-COUNTS-NEXT: Maximum internal block count: 1

// CHECK-COVERAGE: Filename    Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover
// CHECK-COVERAGE-NEXT: ---
// CHECK-COVERAGE-NEXT: basic.c      4                 1    75.00%           1                 0   100.00%           5                 1    80.00%
// CHECK-COVERAGE-NEXT: ---
// CHECK-COVERAGE-NEXT: TOTAL        4                 1    75.00%           1                 0   100.00%           5                 1    80.00%

extern int __llvm_profile_is_continuous_mode_enabled(void);

int main() {
  if (__llvm_profile_is_continuous_mode_enabled())
    return 0;
  return 1;
}
