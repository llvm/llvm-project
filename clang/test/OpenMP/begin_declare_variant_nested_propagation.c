// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s
// expected-no-diagnostics

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
