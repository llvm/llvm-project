// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s
// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// TODO: Issue an eror message as the end is missing
// expected-no-diagnostics
#pragma omp begin declare variant match(device={kind(cpu)})
int also_before(void) {
  return 0;
}

#pragma omp begin declare variant match(device={kind(gpu)})
int also_after(void) {
  return 2;
}
int also_before(void) {
  return 2;
}
#pragma omp end declare variant


#pragma omp begin declare variant match(device={kind(fpga)})

This text is never parsed!

#pragma omp end declare variant

int also_after(void) {
  return 0;
}

int test() {
  return also_after() + also_before();
}
