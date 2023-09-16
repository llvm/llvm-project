// Test debug info correlate with clang coverage.

// Test the case when there is no __llvm_prf_names in the binary.
// RUN: %clang_profgen -o %t.normal -fcoverage-mapping %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: %clang_profgen -o %t -g -mllvm --debug-info-correlate -fcoverage-mapping %S/../Inputs/instrprof-debug-info-correlate-main.cpp %S/../Inputs/instrprof-debug-info-correlate-foo.cpp
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.proflite

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// RUN: llvm-cov report --instr-profile=%t.profdata %t | FileCheck %s -check-prefix=NONAME

// Test debug info correlate with clang coverage (online merging).

// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t.normal
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t-1.profraw %t-2.profraw

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.profdir

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// RUN: llvm-cov report --instr-profile=%t.profdata %t | FileCheck %s -check-prefix=NONAME
// RUN: llvm-cov report --instr-profile=%t.normal.profdata %t | FileCheck %s -check-prefix=NONAME

// NONAME:      Filename                                    Regions    Missed Regions     Cover   Functions  Missed Functions  Executed       Lines      Missed Lines     Cover    Branches   Missed Branches     Cover
// NONAME-NEXT: --
// NONAME-NEXT: instrprof-debug-info-correlate-bar.h              3                 1    66.67%           1                 0   100.00%           5                 1    80.00%           2                 1    50.00%
// NONAME-NEXT: instrprof-debug-info-correlate-foo.cpp            5                 2    60.00%           2                 1    50.00%           6                 2    66.67%           2                 1    50.00%
// NONAME-NEXT: instrprof-debug-info-correlate-main.cpp           4                 0   100.00%           1                 0   100.00%           5                 0   100.00%           2                 0   100.00%
// NONAME-NEXT: --
// NONAME-NEXT: TOTAL                                            12                 3    75.00%           4                 1    75.00%          16                 3    81.25%           6                 2    66.67%

// Test the case when there is __llvm_prf_names in the binary (those are names of uninstrumented functions).
// RUN: %clang_profgen -o %t.normal -fcoverage-mapping -mllvm -enable-name-compression=false %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t.profraw

// RUN: %clang_profgen -o %t -g -mllvm --debug-info-correlate -fcoverage-mapping -mllvm -enable-name-compression=false %s
// RUN: env LLVM_PROFILE_FILE=%t.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.proflite

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// RUN: llvm-cov export --format=lcov --instr-profile=%t.profdata %t | FileCheck %s -check-prefix=NAME

// Test debug info correlate with clang coverage (online merging).

// RUN: env LLVM_PROFILE_FILE=%t-1.profraw %run %t.normal
// RUN: env LLVM_PROFILE_FILE=%t-2.profraw %run %t.normal
// RUN: llvm-profdata merge -o %t.normal.profdata %t-1.profraw %t-2.profraw

// RUN: rm -rf %t.profdir && mkdir %t.profdir
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: env LLVM_PROFILE_FILE=%t.profdir/%m.proflite %run %t
// RUN: llvm-profdata merge -o %t.profdata --debug-info=%t.dSYM %t.profdir

// RUN: diff <(llvm-profdata show --all-functions --counts %t.normal.profdata) <(llvm-profdata show --all-functions --counts %t.profdata)

// RUN: llvm-cov export --format=lcov --instr-profile=%t.profdata %t | FileCheck %s -check-prefix=NAME
// NAME: _Z9used_funcv
// NAME: main
// NAME: _ZN1A11unused_funcEv

struct A {
  void unused_func() {}
};
void used_func() {}
int main() {
  used_func();
  return 0;
}
