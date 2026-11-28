// RUN: %clang_cc1 %s -triple arm64-apple-ios -fptrauth-calls -fexperimental-relative-c++-abi-vtables -emit-llvm -o - | FileCheck %s

// CHECK-LABEL: define{{.*}} void @_Z8call_fooP1A(
// CHECK: [[THIS:%.*]] = load ptr, ptr %a.addr
// CHECK: [[VTABLE:%.*]] = load ptr, ptr [[THIS]]
// CHECK: [[VTABLE_INT:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// CHECK: [[AUTH:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLE_INT]], i32 2, i64 0)
// CHECK: [[VTABLE_AUTH:%.*]] = inttoptr i64 [[AUTH]] to ptr
// CHECK: [[FN:%.*]] = call ptr @llvm.load.relative.i32(ptr [[VTABLE_AUTH]], i32 0)
// CHECK: call void [[FN]](ptr {{.*}}[[THIS]]){{$}}

struct A {
  virtual void foo();
};

void call_foo(A *a) { a->foo(); }
