// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda

#include <stdio.h>
#include <omp.h>

#pragma omp declare target
int isHost;
#pragma omp end declare target

int main(void) {
  isHost = -1;

#pragma omp target enter data map(to: isHost)

#pragma omp target
  { isHost = omp_is_initial_device(); }
#pragma omp target update from(isHost)

  if (isHost < 0) {
    printf("Runtime error, isHost=%d\n", isHost);
  }

#pragma omp target exit data map(delete: isHost)

  // CHECK: Target region executed on the device
  printf("Target region executed on the %s\n", isHost ? "host" : "device");

  return isHost;
}
