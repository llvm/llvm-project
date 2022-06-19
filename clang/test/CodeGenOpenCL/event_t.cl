// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -O0 | FileCheck %s

void foo(event_t evt);

void kernel ker() {
  event_t e;
// CHECK: alloca %opencl.event_t*,
  foo(e);
// CHECK: call {{.*}}void @foo(%opencl.event_t* %
  foo(0);
// CHECK: call {{.*}}void @foo(%opencl.event_t* null)
  foo((event_t)0);
// CHECK: call {{.*}}void @foo(%opencl.event_t* null)
}
