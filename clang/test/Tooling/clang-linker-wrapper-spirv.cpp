// Verify spirv-link flags when called from clang-linker-wrapper
// REQUIRES: system-linux
// REQUIRES: spirv-tools
// REQUIRES: spirv-registered-target
// RUN: %clangxx -fopenmp -fopenmp-targets=spirv64-intel -nogpulib -v -o %t.o %s 2>&1 | FileCheck %s

// CHECK: spirv-link{{.*}} --allow-partial-linkage
int main(int argc, char** argv) {
  return 0;
}
