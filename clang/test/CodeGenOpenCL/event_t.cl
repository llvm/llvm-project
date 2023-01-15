// RUN: %clang_cc1 %s -emit-llvm -o - -O0 | FileCheck %s

void foo(event_t evt);

void kernel ker() {
  event_t e;
// CHECK: alloca ptr,
  foo(e);
// CHECK: call {{.*}}void @foo(ptr %
  foo(0);
// CHECK: call {{.*}}void @foo(ptr null)
  foo((event_t)0);
// CHECK: call {{.*}}void @foo(ptr null)
}
