// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-cir %s -o %t.cir -verify
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-llvm %s -o %t-cir.ll -verify

void HelloWorld(int *A, int *B, int *C, int N) {

// expected-error@+2{{ClangIR code gen Not Yet Implemented: OpenACC Combined Construct}}
// expected-error@+1{{ClangIR code gen Not Yet Implemented: statement}}
#pragma acc parallel loop
  for (unsigned I = 0; I < N; ++I)
    A[I] = B[I] + C[I];

// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Declare Construct}}
#pragma acc declare create(A)
}
