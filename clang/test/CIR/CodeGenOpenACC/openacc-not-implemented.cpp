// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-cir %s -o %t.cir -verify

void HelloWorld(int *A, int *B, int *C, int N) {

// expected-error@+2{{ClangIR code gen Not Yet Implemented: OpenACC Atomic Construct}}
// expected-error@+1{{ClangIR code gen Not Yet Implemented: emitCompoundStmtWithoutScope: OpenACCAtomicConstruct}}
#pragma acc atomic
  N = N + 1;

// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Declare Construct}}
#pragma acc declare create(A)

  // expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Clause: reduction}}
#pragma acc parallel loop reduction(+:A)
  for(int i = 0; i <5; ++i);
}
