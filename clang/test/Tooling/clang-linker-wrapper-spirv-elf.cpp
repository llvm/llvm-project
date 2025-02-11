// Verify the ELF packaging of OpenMP SPIR-V device images.
// REQUIRES: system-linux
// REQUIRES: spirv-tools
// RUN: mkdir -p %t_tmp
// RUN: cd %t_tmp
// RUN: %clangxx -fopenmp -fopenmp-targets=spirv64-intel -nogpulib -c -o %t_clang-linker-wrapper-spirv-elf.o %s
// RUN: not clang-linker-wrapper -o a.out %t_clang-linker-wrapper-spirv-elf.o --save-temps --linker-path=ld
// RUN: clang-offload-packager --image=triple=spirv64-intel,kind=openmp,file=%t.elf  %t_tmp/a.out.openmp.image.wrapper.o
// RUN: llvm-readelf -t %t.elf | FileCheck -check-prefix=CHECK-SECTION %s
// RUN: llvm-readelf -n %t.elf | FileCheck -check-prefix=CHECK-NOTES %s

// CHECK-SECTION: .note.inteloneompoffload
// CHECK-SECTION: __openmp_offload_spirv_0

// CHECK-NOTES-COUNT-3: INTELONEOMPOFFLOAD
int main(int argc, char** argv) {
  return 0;
}
