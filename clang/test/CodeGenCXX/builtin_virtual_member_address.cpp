// RUN: %clang_cc1 -triple arm64e-apple-ios -fptrauth-calls -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s --check-prefixes CHECK,CHECK-AUTH
// RUN: %clang_cc1 -triple arm64-apple-ios -emit-llvm -no-enable-noundef-analysis -o - %s | FileCheck %s --check-prefixes CHECK,CHECK-NOAUTH

struct Base {
  virtual void func1();
  virtual void func2();
  void nonvirt();
};

struct Derived : Base {
  virtual void func1();
};

// CHECK-LABEL: define ptr @_Z7simple1R4Base(ptr{{.*}}%b)

// CHECK-AUTH: [[BLOC:%.*]] = alloca ptr
// CHECK-AUTH: store ptr %b, ptr [[BLOC]]
// CHECK-AUTH: [[B:%.*]] = load ptr, ptr [[BLOC]]
// CHECK-AUTH: [[VTABLE:%.*]] = load ptr, ptr [[B]]
// CHECK-AUTH: [[VTABLE_AUTH_IN:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// CHECK-AUTH: [[VTABLE_AUTH_OUT:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLE_AUTH_IN]], i32 2, i64 0)
// CHECK-AUTH: [[VTABLE:%.*]] = inttoptr i64 [[VTABLE_AUTH_OUT]] to ptr
// CHECK-AUTH: [[FUNC_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 1
// CHECK-AUTH: [[FUNC:%.*]] = load ptr, ptr [[FUNC_ADDR]]
// CHECK-AUTH: [[FUNC_ADDR_I64:%.*]] = ptrtoint ptr [[FUNC_ADDR]] to i64
// CHECK-AUTH: [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[FUNC_ADDR_I64]], i64 25637)
// CHECK-AUTH: [[FUNC_I64:%.*]] = ptrtoint ptr [[FUNC]] to i64
// CHECK-AUTH: [[FUNC_AUTHED:%.*]] =  call i64 @llvm.ptrauth.auth(i64 [[FUNC_I64]], i32 0, i64 [[DISC]])
// CHECK-AUTH: [[FUNC:%.*]] = inttoptr i64 [[FUNC_AUTHED]] to ptr
// CHECK-AUTH: ret ptr [[FUNC]]

// CHECK-NOAUTH: [[BLOC:%.*]] = alloca ptr
// CHECK-NOAUTH: store ptr %b, ptr [[BLOC]]
// CHECK-NOAUTH: [[B:%.*]] = load ptr, ptr [[BLOC]]
// CHECK-NOAUTH: [[VTABLE:%.*]] = load ptr, ptr [[B]]
// CHECK-NOAUTH: [[FUNC_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 1
// CHECK-NOAUTH: [[FUNC:%.*]] = load ptr, ptr [[FUNC_ADDR]]
// CHECK-NOAUTH: ret ptr [[FUNC]]
void *simple1(Base &b) {
  return __builtin_virtual_member_address(b, &Base::func2);
}

// CHECK-LABEL: define ptr @_Z7simple2P4Base(ptr{{.*}}%b)

// CHECK-AUTH:   [[B_ADDR:%.*]] = alloca ptr, align 8
// CHECK-AUTH:   store ptr %b, ptr [[B_ADDR]]
// CHECK-AUTH:   [[VTABLE:%.*]] = load ptr, ptr [[B_ADDR]]
// CHECK-AUTH:   [[VTABLE_I64:%.*]] = ptrtoint ptr [[VTABLE]] to i64
// CHECK-AUTH:   [[VTABLE_I64_AUTHED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLE_I64]], i32 2, i64 0)
// CHECK-AUTH:   [[VTABLE_AUTHED:%.*]] = inttoptr i64 [[VTABLE_I64_AUTHED]] to ptr
// CHECK-AUTH:   [[VFUNCTION_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE_AUTHED]], i64 1
// CHECK-AUTH:   [[VFUNCTION_PTR:%.*]] = load ptr, ptr [[VFUNCTION_ADDR]]
// CHECK-AUTH:   [[VFUNCTION_ADDR_I64:%.*]] = ptrtoint ptr [[VFUNCTION_ADDR]] to i64
// CHECK-AUTH:   [[DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VFUNCTION_ADDR_I64]], i64 25637)
// CHECK-AUTH:   [[VFUNCTION_I64:%.*]] = ptrtoint ptr [[VFUNCTION_PTR]] to i64
// CHECK-AUTH:   [[AUTHED_FPTR_I64:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VFUNCTION_I64]], i32 0, i64 %5)
// CHECK-AUTH:   [[AUTHED_FPTR:%.*]] = inttoptr i64 [[AUTHED_FPTR_I64]] to ptr
// CHECK-AUTH:   ret ptr [[AUTHED_FPTR]]

// CHECK-NOAUTH: [[B_ADDR:%.*]] = alloca ptr
// CHECK-NOAUTH: store ptr %b, ptr [[B_ADDR]]
// CHECK-NOAUTH: [[VTABLE:%.*]] = load ptr, ptr [[B_ADDR]]
// CHECK-NOAUTH: [[FUNC_ADDR:%.*]] = getelementptr inbounds ptr, ptr [[VTABLE]], i64 1
// CHECK-NOAUTH: [[FUNC:%.*]] = load ptr, ptr [[FUNC_ADDR]]
// CHECK-NOAUTH: ret ptr [[FUNC]]
void *simple2(Base *b) {
  return __builtin_virtual_member_address(b, &Base::func2);
}
