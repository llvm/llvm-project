// REQUIRES: osx-ld64-live_support
// REQUIRES: lto

// RUN: %clang_profgen=%t.profraw -fcoverage-mapping -mllvm -enable-name-compression=false -Wl,-dead_strip -o %t %s
// RUN: %run %t
// RUN: llvm-profdata merge -o %t.profdata %t.profraw
// RUN: llvm-profdata show --all-functions %t.profdata | FileCheck %s -check-prefix=PROF
// RUN: llvm-cov show %t -instr-profile %t.profdata | FileCheck %s -check-prefix=COV
// RUN: nm %t | FileCheck %s -check-prefix=NM
// RUN: otool -s __DATA __llvm_prf_names %t | FileCheck %s -check-prefix=PRF_NAMES
// RUN: otool -s __DATA __llvm_prf_cnts %t | FileCheck %s -check-prefix=PRF_CNTS

// RUN: %clang_lto_profgen=%t.lto.profraw -fcoverage-mapping -mllvm -enable-name-compression=false -Wl,-dead_strip -flto -o %t.lto %s
// RUN: %run %t.lto
// RUN: llvm-profdata merge -o %t.lto.profdata %t.lto.profraw
// RUN: llvm-profdata show --all-functions %t.lto.profdata | FileCheck %s -check-prefix=PROF
// RUN: llvm-cov show %t.lto -instr-profile %t.lto.profdata | FileCheck %s -check-prefix=COV
// RUN: nm %t.lto | FileCheck %s -check-prefix=NM
// RUN: otool -s __DATA __llvm_prf_names %t.lto | FileCheck %s -check-prefix=PRF_NAMES
// RUN: otool -s __DATA __llvm_prf_cnts %t.lto | FileCheck %s -check-prefix=PRF_CNTS

// Note: We expect foo() and some of the profiling data associated with it to
// be dead-stripped.

// COV: [[@LINE+1]]{{ *}}|{{ *}}0|void foo()
void foo() {}

// COV: [[@LINE+1]]{{ *}}|{{ *}}1|int main
int main() { return 0; }

// NM-NOT: foo

// PROF: Counters:
// PROF-NEXT:   main:
// PROF-NEXT:     Hash:
// PROF-NEXT:     Counters: 1
// PROF-NEXT:     Function count: 1
// PROF-NEXT: Instrumentation level: Front-end
// PROF-NEXT: Functions shown: 1
// PROF-NEXT: Total functions: 1
// PROF-NEXT: Maximum function count: 1
// PROF-NEXT: Maximum internal block count: 0

// Note: We don't expect the names of dead-stripped functions to disappear from
// __llvm_prf_names, because collectPGOFuncNameStrings() glues the names
// together.

// PRF_NAMES: Contents of (__DATA,__llvm_prf_names) section
// PRF_NAMES-NEXT: {{.*}} 08 00 66 6f 6f 01 6d 61 69 6e{{ +$}}
//                        |  |  f  o  o  #  m  a  i  n
//                        |  |___________|
//                        |              |
//               UncompressedLen = 8     |
//                                       |
//                                CompressedLen = 0

// Note: We expect the profile counters for dead-stripped functions to also be
// dead-stripped.

// PRF_CNTS: Contents of (__DATA,__llvm_prf_cnts) section
// PRF_CNTS-NEXT: {{.*}} 00 00 00 00 00 00 00 00{{ +$}}
