// RUN: %clang -### --target=powerpc-ibm-aix-xcoff -mno-inline-glue %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=NO_INLINE_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mno-inline-glue %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=NO_INLINE_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -minline-glue %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=INLINE_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=INLINE_GLUE

// RUN: %clang -### --target=powerpc64-ibm-aix-xcoff -mno-inline-glue -minline-glue %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=INLINE_GLUE

// RUN: not %clang -### --target=powerpc64le-unknown-linux-gnu -mno-inline-glue \
// RUN:     %s 2>&1 | FileCheck %s --check-prefix=ERR

// RUN: %clang -target powerpc-unkown-aix -mno-inline-glue %s -S -emit-llvm -o - | \
// RUN:     FileCheck %s

// RUN: %clang -target powerpc-unkown-aix -mno-inline-glue -minline-glue %s -S -emit-llvm -o - | \
// RUN:     FileCheck %s --check-prefix=DIS

// NO_INLINE_GLUE: "-target-feature" "+use-ptrgl-helper"
// INLINE_GLUE-NOT: "+use-ptrgl-helper"
// ERR: error: unsupported option '-mno-inline-glue' for target 'powerpc64le-unknown-linux-gnu'

int test(void) {
  return 0;
}

// CHECK: test() #0 {
// CHECK: attributes #0 = {
// CHECK-ON-SAME: +use-ptrgl-helper

// DIS-NOT: +use-ptrgl-helper
