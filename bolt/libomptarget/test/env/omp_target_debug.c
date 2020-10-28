// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 | %fcheck-aarch64-unknown-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu && env LIBOMPTARGET_DEBUG=0 %libomptarget-run-aarch64-unknown-linux-gnu 2>&1 | %fcheck-aarch64-unknown-linux-gnu -allow-empty -check-prefix=NDEBUG
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 | %fcheck-powerpc64-ibm-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=0 %libomptarget-run-powerpc64-ibm-linux-gnu 2>&1 | %fcheck-powerpc64-ibm-linux-gnu -allow-empty -check-prefix=NDEBUG
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 | %fcheck-powerpc64le-ibm-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu && env LIBOMPTARGET_DEBUG=0 %libomptarget-run-powerpc64le-ibm-linux-gnu 2>&1 | %fcheck-powerpc64le-ibm-linux-gnu -allow-empty -check-prefix=NDEBUG
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-x86_64-pc-linux-gnu 2>&1 | %fcheck-x86_64-pc-linux-gnu -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-x86_64-pc-linux-gnu && env LIBOMPTARGET_DEBUG=0 %libomptarget-run-x86_64-pc-linux-gnu 2>&1 | %fcheck-x86_64-pc-linux-gnu -allow-empty -check-prefix=NDEBUG
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda && env LIBOMPTARGET_DEBUG=1 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda -allow-empty -check-prefix=DEBUG
// RUN: %libomptarget-compile-nvptx64-nvidia-cuda && env LIBOMPTARGET_DEBUG=0 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda -allow-empty -check-prefix=NDEBUG
// REQUIRES: libomptarget-debug

int main(void) {
#pragma omp target
  {}
  return 0;
}

// DEBUG: Libomptarget
// NDEBUG-NOT: Libomptarget
// NDEBUG-NOT: Target

