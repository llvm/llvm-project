// Verify the ELF packaging of OpenMP SPIR-V device images.
// REQUIRES: system-linux
// REQUIRES: spirv-tools
// REQUIRES: spirv-registered-target
// RUN: %clangxx -fopenmp -fopenmp-targets=spirv64-intel -nogpulib -o %t %s
// RUN: llvm-objdump --offloading %t | FileCheck -check-prefix=CHECK %s

// CHECK: [Nested OffloadBinary
// CHECK: Number of inner images: 1
// CHECK: spirv64-intel

int main(int argc, char** argv) {
  return 0;
}
