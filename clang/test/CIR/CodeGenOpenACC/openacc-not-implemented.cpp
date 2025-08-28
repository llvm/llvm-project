// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-cir %s -o %t.cir -verify

void HelloWorld(int *A, int *B, int *C, int N) {

// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Atomic Construct}}
#pragma acc atomic
  N = N + 1;

// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Declare Construct}}
#pragma acc declare create(A)
}
