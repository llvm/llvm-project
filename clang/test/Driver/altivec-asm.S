// RUN: %clang -target powerpc64-linux-gnu -maltivec -S %s -o - | FileCheck %s
// RUN: %clang -target powerpc64le-linux-gnu -maltivec -S %s -o - | FileCheck %s
// Verify that assembling an empty file does not auto-include altivec.h.
// CHECK-NOT: static vector
