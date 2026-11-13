// RUN: %clang_cc1 -std=gnu99 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

void foo(void);

// CHECK-LABEL: define{{.*}} void @gh215454(
// CHECK: call void @foo()
void gh215454(int e) {
  ({ foo(); __typeof__(e); });
}
