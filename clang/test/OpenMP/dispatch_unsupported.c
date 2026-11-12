// RUN: %clang_cc1 -emit-llvm -fopenmp -disable-llvm-passes %s -o /dev/null -verify=expected

// expected-error@+2 {{cannot compile this OpenMP dispatch directive yet}}
void a(){
    #pragma omp dispatch
    a();
}
