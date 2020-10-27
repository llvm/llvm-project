// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 | %fcheck-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 | %fcheck-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 | %fcheck-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-x86_64-pc-linux-gnu 2>&1 | %fcheck-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda
// REQUIRES: libomptarget-debug

#include <stdlib.h>

int *allocate(size_t n) {
  int *ptr = malloc(sizeof(int) * n);
#pragma omp target enter data map(to : ptr[:n])
  return ptr;
}

void deallocate(int *ptr, size_t n) {
#pragma omp target exit data map(delete : ptr[:n])
  free(ptr);
}

#pragma omp declare target
int *cnt;
void foo() {
  ++(*cnt);
}
#pragma omp end declare target

int main(void) {
  int *A = allocate(10);
  int *V = allocate(10);
  deallocate(A, 10);
  deallocate(V, 10);
// CHECK-NOT: RefCount=2
  cnt = malloc(sizeof(int));
  *cnt = 0;
#pragma omp target map(cnt[:1])
  foo();
  printf("Cnt = %d.\n", *cnt);
// CHECK: Cnt = 1.
  *cnt = 0;
#pragma omp target data map(cnt[:1])
#pragma omp target
  foo();
  printf("Cnt = %d.\n", *cnt);
// CHECK: Cnt = 1.
  free(cnt);

  return 0;
}


