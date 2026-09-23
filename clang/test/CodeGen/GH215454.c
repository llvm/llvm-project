// RUN: %clang_cc1 -std=gnu99 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// A declaration that declares nothing made the enclosing statement expression
// invalid without an error, and the call in it was dropped.

void foo(void);

// CHECK-LABEL: define{{.*}} void @keeps_side_effects(
// CHECK: call void @foo()
void keeps_side_effects(int e) {
  ({ foo(); __typeof__(e); });
}
