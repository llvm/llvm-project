// RUN: %clangxx_profgen -fcoverage-mapping %S/Inputs/instrprof-visibility-helper.cpp -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t
// RUN: llvm-profdata merge %t.profraw -o %t.profdata
// RUN: llvm-profdata show --all-functions %t.profraw | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=PROFILE
// RUN: llvm-cov show %t -instr-profile=%t.profdata | FileCheck -allow-deprecated-dag-overlap %s --check-prefix=COV

namespace {
#define NO_WEAK
#define NO_EXTERN
#include "instrprof-visibility-kinds.inc"
#undef NO_EXTERN
#undef NO_WEAK
}

namespace N1 {
#include "instrprof-visibility-kinds.inc"
}

int main() {
  call();
  N1::call();
  return 0;
}

// PROFILE-DAG: _ZN2N12f1Ev
// PROFILE-DAG: _ZN2N12f2Ev
// PROFILE-DAG: _ZN2N12f3Ev
// PROFILE-DAG: _ZN2N12f4Ev
// PROFILE-DAG: _ZN2N12f5Ev
// PROFILE-DAG: _ZN2N12f6Ev
// PROFILE-DAG: _ZN2N12f7Ev
// PROFILE-DAG: _ZN2N14callEv
// PROFILE-DAG: main
// PROFILE-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_14callEv
// PROFILE-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f1Ev
// PROFILE-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f3Ev
// PROFILE-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f5Ev
// PROFILE-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f6Ev
// PROFILE-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f7Ev
// PROFILE-DAG: Total functions: 15

// COV-DAG: instrprof-visibility-helper.cpp

// COV-DAG: instrprof-visibility-kinds.inc

// COV-DAG: _ZN2N12f1Ev
// COV-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f1Ev
// COV-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f3Ev
// COV-DAG: _ZN2N12f3Ev
// COV-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f5Ev
// COV-DAG: _ZN2N12f5Ev
// COV-DAG: _ZN2N12f6Ev
// COV-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f6Ev
// COV-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_12f7Ev
// COV-DAG: _ZN2N12f7Ev

// --- Check coverage for functions in the anonymous namespace.
// COV-DAG: instrprof-visibility.cpp:_ZN12_GLOBAL__N_14callEv
// COV-DAG: [[CALL:[0-9]+]]|{{ *}}1|void call() {
// COV-DAG: {{.*}}|{{ *}}1|  f1();
// COV-DAG: {{.*}}|{{ *}}1|#ifndef NO_WEAK
// COV-DAG: {{.*}}|{{ *}} |  f2();
// COV-DAG: {{.*}}|{{ *}} |#endif
// COV-DAG: {{.*}}|{{ *}}1|  f3();
// COV-DAG: {{.*}}|{{ *}}1|#ifndef NO_EXTERN
// COV-DAG: {{.*}}|{{ *}} |  f4();
// COV-DAG: {{.*}}|{{ *}} |#endif
// COV-DAG: {{.*}}|{{ *}}1|  f5();
// COV-DAG: {{.*}}|{{ *}}1|  f6();
// COV-DAG: {{.*}}|{{ *}}1|  f7();
// COV-DAG: {{.*}}|{{ *}}1|}

// --- Check coverage for functions in namespace N1.
// COV-DAG: _ZN2N14callEv
// COV-DAG: {{ *}}[[CALL]]|{{ *}}1|void call() {
// COV-DAG: {{.*}}|{{ *}}1|  f1();
// COV-DAG: {{.*}}|{{ *}}1|#ifndef NO_WEAK
// COV-DAG: {{.*}}|{{ *}}1|  f2();
// COV-DAG: {{.*}}|{{ *}}1|#endif
// COV-DAG: {{.*}}|{{ *}}1|  f3();
// COV-DAG: {{.*}}|{{ *}}1|#ifndef NO_EXTERN
// COV-DAG: {{.*}}|{{ *}}1|  f4();
// COV-DAG: {{.*}}|{{ *}}1|#endif
// COV-DAG: {{.*}}|{{ *}}1|  f5();
// COV-DAG: {{.*}}|{{ *}}1|  f6();
// COV-DAG: {{.*}}|{{ *}}1|  f7();
// COV-DAG: {{.*}}|{{ *}}1|}

// COV-DAG: instrprof-visibility.cpp
