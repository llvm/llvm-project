// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 | %fcheck-aarch64-unknown-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 | %fcheck-powerpc64-ibm-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 | %fcheck-powerpc64le-ibm-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-x86_64-pc-linux-gnu 2>&1 | %fcheck-x86_64-pc-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda -allow-empty -check-prefix=DEBUG
// REQUIRES: libomptarget-debug

/*
  Test for looptripcount being popped from runtime stack.
*/
#include <stdio.h>
#include <omp.h>
int main()
{
  int N = 128;
  int NN = 1024;
  int num_teams[NN];
  int num_threads[NN];

  printf("#pragma omp target teams distribute parallel for thread_limit(4)\n");
#pragma omp target teams distribute parallel for thread_limit(4)
  for (int j = 0; j< N; j++) {
    num_threads[j] = omp_get_num_threads();
    num_teams[j] = omp_get_num_teams();
  }
  printf("num_threads %d num_teams %d\n", num_threads[0], num_teams[0]);
// DEBUG: loop trip count is 128
  printf("#pragma omp target teams distribute parallel for\n");
#pragma omp target teams distribute parallel for
  for (int j = 0; j< N; j++) {
    num_threads[j] = omp_get_num_threads();
    num_teams[j] = omp_get_num_teams();
  }
  printf("num_threads %d num_teams %d\n", num_threads[0], num_teams[0]);
// DEBUG: loop trip count is 128
  return 0;
}
