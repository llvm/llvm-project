// RUN: %clang_cc1 -triple powerpc-ibm-aix -target-feature +use-ptrgl-helper \
// RUN:   %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -triple powerpc64-ibm-aix -target-feature +use-ptrgl-helper \
// RUN:   %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefix=DIS

int test(void) {
  return 0;
}

// CHECK: test() #0 {
// CHECK: attributes #0 = {
// CHECK-SAME: "target-features"={{"|"[^"]*,}}+use-ptrgl-helper{{"|,[^"]*"}}

// DIS-NOT: +use-ptrgl-helper
