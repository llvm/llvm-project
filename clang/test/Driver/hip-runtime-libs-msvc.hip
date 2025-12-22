// RUN:  touch %t.o

// Test HIP runtime lib args specified by --rocm-path.
// RUN: %clang -### --hip-link --target=x86_64-pc-windows-msvc \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck %s

// Test HIP runtime lib is linked without --hip-link when there is HIP input file.
// RUN: %clang -### --target=x86_64-pc-windows-msvc -nogpuinc -nogpulib \
// RUN:   --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck %s

// Test HIP runtime lib is linked even if -nostdlib is specified when the input
// is a HIP file. This is important when composing with e.g. the UCRT or other
// non glibc-like implementations of the C standard library.
// RUN: %clang -### --target=x86_64-pc-windows-msvc -nogpuinc -nogpulib \
// RUN:   -nostdlib --rocm-path=%S/Inputs/rocm %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-libpath:{{.*Inputs.*rocm.*lib}}" "amdhip64.lib"
