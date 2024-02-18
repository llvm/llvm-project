// REQUIRES: arm-registered-target

// RUN: %clang -cc1as -triple armv7-unknown-linuxfdpiceabi -filetype obj --fdpic %s -o %t
// RUN: llvm-readelf -h -r %t | FileCheck %s

// CHECK: OS/ABI: ARM FDPIC
// CHECK: R_ARM_FUNCDESC

.data
.word f(FUNCDESC)
