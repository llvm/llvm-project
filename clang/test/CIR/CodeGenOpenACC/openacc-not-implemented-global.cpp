// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-cir %s -o %t.cir -verify
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fopenacc -fclangir -emit-llvm %s -o %t-cir.ll -verify

int Global;
// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenACC Declare Construct}}
#pragma acc declare create(Global)
