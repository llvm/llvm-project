// RUN: %clang_cc1 %s -triple amdgpu-amd-amdhsa -emit-llvm -std=c++11 -O1 -o - | FileCheck %s \
// RUN:   --implicit-check-not='call {{.*}} @__dynamic_cast'

struct A {
  virtual ~A();
};
struct B final : A {};

// CHECK-LABEL: define {{.*}} ptr @_Z4castP1A(
// CHECK: %[[VTABLE:.*]] = load ptr addrspace(1), ptr %{{.*}}
// CHECK: %[[MATCH:.*]] = icmp eq ptr addrspace(1) %[[VTABLE]], getelementptr {{.*}} ptr addrspace(1) @_ZTV1B
B *cast(A *a) {
  return dynamic_cast<B *>(a);
}

struct Left : A {};
struct Right : A {};
struct Repeated final : Left, Right {};

// CHECK-LABEL: define {{.*}} ptr @_Z13cast_repeatedP1A(
// CHECK: %[[PRIMARY:.*]] = getelementptr inbounds i8, ptr %{{.*}}, i64 %{{.*}}
// CHECK: %[[VTABLE:.*]] = load ptr addrspace(1), ptr %[[PRIMARY]]
// CHECK: %[[MATCH:.*]] = icmp eq ptr addrspace(1) %[[VTABLE]], getelementptr {{.*}} ptr addrspace(1) @_ZTV8Repeated
Repeated *cast_repeated(A *a) {
  return dynamic_cast<Repeated *>(a);
}
