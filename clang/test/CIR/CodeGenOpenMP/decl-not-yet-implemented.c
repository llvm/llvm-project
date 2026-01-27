// RUN: %clang_cc1 -fopenmp -fclangir %s -verify -emit-cir -o -

int a;
// expected-error@+1{{ClangIR code gen Not Yet Implemented: OpenMP OMPThreadPrivateDecl}}
#pragma omp threadprivate(a)
