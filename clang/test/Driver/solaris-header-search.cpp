// Test that the C++ headers are found.
//
// RUN: %clang -no-canonical-prefixes %s -### 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/sparc-sun-solaris2.11 \
// RUN:   | FileCheck %s
// CHECK: "{{[^"]*}}clang{{[^"]*}}" "-cc1"
// CHECK: "-internal-isystem" "{{.*}}/usr/include/c++/v1/support/solaris"
// CHECK: "-internal-isystem" "{{.*}}/usr/gcc/4.8/include/c++/4.8.2"
// CHECK: "-internal-isystem" "{{.*}}/usr/gcc/4.8/include/c++/4.8.2/sparc-sun-solaris2.11"
