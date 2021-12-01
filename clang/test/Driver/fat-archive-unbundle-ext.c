// REQUIRES: x86-registered-target
// UNSUPPORTED: windows, darwin, aix

// Generate dummy fat object
// RUN: %clang -O0 -target %itanium_abi_triple %s -c -o %t.host.o
// RUN: echo 'Content of device file' > %t.tgt.o
// RUN: clang-offload-bundler -type=o -targets=host-%itanium_abi_triple,openmp-%itanium_abi_triple -inputs=%t.host.o,%t.tgt.o -outputs=%t.fat.obj

// Then create a static archive with that object
// RUN: rm -f %t.fat.a
// RUN: llvm-ar cr %t.fat.a %t.fat.obj

// Unbundle device part from the archive. Check that bundler does not print warnings.
// RUN: clang-offload-bundler -unbundle -type=a -targets=openmp-%itanium_abi_triple -inputs=%t.fat.a -outputs=%t.tgt.a 2>&1 | FileCheck --allow-empty --check-prefix=CHECK-WARNING %s
// CHECK-WARNING-NOT: warning

// Check that device archive member inherited file extension from the original file.
// RUN: llvm-ar t %t.tgt.a | FileCheck --check-prefix=CHECK-ARCHIVE %s
// CHECK-ARCHIVE: {{\.obj$}}

void foo(void) {}
