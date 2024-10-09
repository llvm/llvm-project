// RUN: %clang_cc1 %s -emit-llvm -O2 -fextend-lifetimes -o - | FileCheck --check-prefixes=CHECK-ALL,CHECK-O2 %s
// RUN: %clang_cc1 %s -emit-llvm -O0 -fextend-lifetimes -o - | FileCheck --check-prefixes=CHECK-ALL,CHECK-O0 %s

// Checks that we emit the function attribute has_fake_uses when
// -fextend-lifetimes is on and optimizations are enabled, and that it does not
// when optimizations are disabled.

// CHECK-ALL:    define {{.*}}void @foo
// CHECK-O2:     attributes #0 = {{{.*}}has_fake_uses
// CHECK-O0-NOT: attributes #0 = {{{.*}}has_fake_uses

void foo() {}
