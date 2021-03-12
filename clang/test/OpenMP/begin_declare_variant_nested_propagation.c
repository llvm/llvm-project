// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

#pragma omp begin declare variant match(user={condition(1)}, device={kind(cpu)}, implementation={extension(match_any)})
#pragma omp begin declare variant match(device = {kind(cpu, fpga)})
 this is never reached
#pragma omp end declare variant
#pragma omp end declare variant

#pragma omp begin declare variant match(user={condition(1)}, device={kind(cpu)}, implementation={extension(match_any)})
#pragma omp begin declare variant match(device = {kind(cpu, fpga)}, implementation={vendor(llvm)})
 this is never reached
#pragma omp end declare variant
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation={extension(disable_implicit_base, disable_selector_propagation)})

 void without_implicit_base() {}

#pragma omp begin declare variant match(implementation = {vendor(llvm)})
 void with_implicit_base() {}
#pragma omp end declare variant

#pragma omp end declare variant

 void test() {
   without_implicit_base(); // expected-warning{{implicit declaration of function 'without_implicit_base' is invalid in C99}}
   with_implicit_base();
 }
