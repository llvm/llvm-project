// REQUIRES: system-windows

// RUN:  touch %t.o

// Test HIP runtime lib args specified by --rocm-path.
// RUN: %clang -### --hip-link -target x86_64-pc-windows-msvc \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-libpath:{{.*Inputs.*rocm.*lib}}" "amdhip64.lib"
