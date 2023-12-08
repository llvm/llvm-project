// Test that the montavista gcc-toolchain is correctly detected
//
// RUN: %clang -rtlib=platform -print-libgcc-file-name 2>&1 \
// RUN:     --target=i686-montavista-linux \
// RUN:     --gcc-toolchain=%S/Inputs/montavista_i686_tree/usr \
// RUN:   | FileCheck %s

// Test for header search toolchain detection.
// CHECK: montavista_i686_tree/usr/lib/gcc/i686-montavista-linux/4.2.0{{/|\\}}libgcc.a
