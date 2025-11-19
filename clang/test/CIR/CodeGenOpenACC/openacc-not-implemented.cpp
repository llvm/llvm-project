// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-cir %s -o %t.cir -verify

void HelloWorld(int *A) {

// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Declare Construct}}
#pragma acc declare create(A)
}
