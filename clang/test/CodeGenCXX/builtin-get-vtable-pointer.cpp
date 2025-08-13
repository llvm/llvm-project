// RUN: %clang_cc1 %s -x c++ -std=c++23  -triple x86_64-apple-darwin10 -emit-llvm -O1 -disable-llvm-passes -no-enable-noundef-analysis  -o - | FileCheck --check-prefix=CHECK-NOAUTH %s
// RUN: %clang_cc1 %s -x c++ -std=c++23  -triple arm64-apple-ios -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -emit-llvm -O1 -disable-llvm-passes -no-enable-noundef-analysis   -o - | FileCheck --check-prefix=CHECK-TYPEAUTH %s
// RUN: %clang_cc1 %s -x c++ -std=c++23  -triple arm64-apple-ios -fptrauth-calls -fptrauth-vtable-pointer-address-discrimination -emit-llvm -O1 -disable-llvm-passes -no-enable-noundef-analysis  -o - | FileCheck --check-prefix=CHECK-ADDRESSAUTH %s
// RUN: %clang_cc1 %s -x c++ -std=c++23  -triple arm64-apple-ios -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-address-discrimination -emit-llvm -O1 -disable-llvm-passes -no-enable-noundef-analysis  -o - | FileCheck --check-prefix=CHECK-BOTHAUTH %s
// FIXME: Assume load should not require -fstrict-vtable-pointers

namespace test1 {
struct A {
  A();
  virtual void bar();
};

struct B : A {
  B();
  virtual void foo();
};

struct Z : A {};
struct C : Z, B {
  C();
  virtual void wibble();
};

struct D : virtual A {
};

struct E : D, B {
};

template <class A, class B> struct same_type {
  static const bool value = false;
};

template <class A> struct same_type<A, A> {
  static const bool value = true;
};

const void *a(A *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test11aEPNS_1AE(ptr %o) #0 {
  // CHECK-TYPEAUTH: define ptr @_ZN5test11aEPNS_1AE(ptr %o) #0 {
  return __builtin_get_vtable_pointer(o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %0 = load ptr, ptr %o.addr, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 %1)
  // CHECK-ADDRESSAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-ADDRESSAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-BOTHAUTH: [[T1:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T1]], i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *b(B *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-TYPEAUTH: define ptr @_ZN5test11bEPNS_1BE(ptr %o) #0 {
  // CHECK-NOAUTH: define ptr @_ZN5test11bEPNS_1BE(ptr %o) #0 {
  return __builtin_get_vtable_pointer(o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 %1)
  // CHECK-ADDRESSAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-ADDRESSAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 %1, i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *b_as_A(B *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test16b_as_AEPNS_1BE(ptr %o) #0 {
  return __builtin_get_vtable_pointer((A *)o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 %1)
  // CHECK-ADDRESSAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-ADDRESSAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 %1, i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *c(C *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test11cEPNS_1CE(ptr %o) #0 {
  return __builtin_get_vtable_pointer(o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 %1)
  // CHECK-ADDRESSAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-ADDRESSAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 %1, i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *c_as_Z(C *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test16c_as_ZEPNS_1CE(ptr %o) #0 {
  return __builtin_get_vtable_pointer((Z *)o);
  // CHECK-NOAUTH: %0 = load ptr, ptr %o.addr, align 8
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 %1)
  // CHECK-ADDRESSAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-ADDRESSAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 %1, i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *c_as_B(C *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test16c_as_BEPNS_1CE(ptr %o) #0 {
  return __builtin_get_vtable_pointer((B *)o);
  // CHECK-NOAUTH: %add.ptr = getelementptr inbounds i8, ptr %0, i64 8
  // CHECK-NOAUTH: br label %cast.end
  // CHECK-NOAUTH: %cast.result = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ]
  // CHECK-NOAUTH: %vtable = load ptr, ptr %cast.result, align 8
  // CHECK-TYPEAUTH: %cast.result = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ]
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %cast.result, align 8
  // CHECK-TYPEAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-TYPEAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %cast.result to i64
  // CHECK-ADDRESSAUTH: %3 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %4 = call i64 @llvm.ptrauth.auth(i64 %3, i32 2, i64 %2)
  // CHECK-ADDRESSAUTH: %5 = inttoptr i64 %4 to ptr
  // CHECK-ADDRESSAUTH: %6 = load volatile i8, ptr %5, align 8
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 %2, i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *d(D *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test11dEPNS_1DE(ptr %o) #0 {
  return __builtin_get_vtable_pointer(o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: %1 = ptrtoint ptr %0 to i64
  // CHECK-ADDRESSAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 %1)
  // CHECK-ADDRESSAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-ADDRESSAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-BOTHAUTH: [[T1:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T1]], i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *d_as_A(D *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test16d_as_AEPNS_1DE(ptr %o) #0 {
  return __builtin_get_vtable_pointer((A *)o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-NOAUTH: %vbase.offset.ptr = getelementptr i8, ptr %vtable, i64 -32
  // CHECK-NOAUTH: %vbase.offset = load i64, ptr %vbase.offset.ptr, align 8
  // CHECK-NOAUTH: %add.ptr = getelementptr inbounds i8, ptr %0, i64 %vbase.offset
  // CHECK-NOAUTH: %cast.result = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ]
  // CHECK-NOAUTH: %vtable1 = load ptr, ptr %cast.result, align 8
  // CHECK-TYPEAUTH: %vtable1 = load ptr, ptr %cast.result, align 8
  // CHECK-TYPEAUTH: %5 = ptrtoint ptr %vtable1 to i64
  // CHECK-TYPEAUTH: %6 = call i64 @llvm.ptrauth.auth(i64 %5, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %7 = inttoptr i64 %6 to ptr
  // CHECK-TYPEAUTH: %8 = load volatile i8, ptr %7, align 8
  // CHECK-ADDRESSAUTH: %6 = ptrtoint ptr %cast.result to i64
  // CHECK-ADDRESSAUTH: %7 = ptrtoint ptr %vtable1 to i64
  // CHECK-ADDRESSAUTH: %8 = call i64 @llvm.ptrauth.auth(i64 %7, i32 2, i64 %6)
  // CHECK-ADDRESSAUTH: %9 = inttoptr i64 %8 to ptr
  // CHECK-ADDRESSAUTH: %10 = load volatile i8, ptr %9, align 8
  // CHECK-BOTHAUTH: [[T1:%.*]] = ptrtoint ptr %cast.result to i64
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T1]], i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable1 to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *e(E *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test11eEPNS_1EE(ptr %o) #0 {
  return __builtin_get_vtable_pointer(o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: [[T1:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-ADDRESSAUTH: [[T2:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: [[T3:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T2]], i32 2, i64 [[T1]])
  // CHECK-ADDRESSAUTH: [[T4:%.*]] = inttoptr i64 [[T3]] to ptr
  // CHECK-ADDRESSAUTH: [[T5:%.*]] = load volatile i8, ptr [[T4]], align 8
  // CHECK-BOTHAUTH: [[T1:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T1]], i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *e_as_B(E *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test16e_as_BEPNS_1EE(ptr %o) #0 {
  return __builtin_get_vtable_pointer((B *)o);
  // CHECK-NOAUTH: %add.ptr = getelementptr inbounds i8, ptr %0, i64 8
  // CHECK-NOAUTH: %cast.result = phi ptr [ %add.ptr, %cast.notnull ], [ null, %entry ]
  // CHECK-NOAUTH: %vtable = load ptr, ptr %cast.result, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %cast.result, align 8
  // CHECK-TYPEAUTH: %2 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %3 = call i64 @llvm.ptrauth.auth(i64 %2, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %4 = inttoptr i64 %3 to ptr
  // CHECK-TYPEAUTH: %5 = load volatile i8, ptr %4, align 8
  // CHECK-ADDRESSAUTH: [[T1:%.*]] = ptrtoint ptr %cast.result to i64
  // CHECK-ADDRESSAUTH: [[T2:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: [[T3:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T2]], i32 2, i64 [[T1]])
  // CHECK-ADDRESSAUTH: [[T4:%.*]] = inttoptr i64 [[T3]] to ptr
  // CHECK-ADDRESSAUTH: [[T5:%.*]] = load volatile i8, ptr [[T4]], align 8
  // CHECK-BOTHAUTH: [[T1:%.*]] = ptrtoint ptr %cast.result to i64
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T1]], i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

const void *e_as_D(E *o) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(o)), const void*>::value);
  // CHECK-NOAUTH: define ptr @_ZN5test16e_as_DEPNS_1EE(ptr %o) #0 {
  return __builtin_get_vtable_pointer((D *)o);
  // CHECK-NOAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %0, align 8
  // CHECK-TYPEAUTH: %1 = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: %2 = call i64 @llvm.ptrauth.auth(i64 %1, i32 2, i64 48388)
  // CHECK-TYPEAUTH: %3 = inttoptr i64 %2 to ptr
  // CHECK-TYPEAUTH: %4 = load volatile i8, ptr %3, align 8
  // CHECK-ADDRESSAUTH: [[T1:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-ADDRESSAUTH: [[T2:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: [[T3:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T2]], i32 2, i64 [[T1]])
  // CHECK-ADDRESSAUTH: [[T4:%.*]] = inttoptr i64 [[T3]] to ptr
  // CHECK-ADDRESSAUTH: [[T5:%.*]] = load volatile i8, ptr [[T4]], align 8
  // CHECK-BOTHAUTH: [[T1:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-BOTHAUTH: [[T2:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[T1]], i64 48388)
  // CHECK-BOTHAUTH: [[T3:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[T4:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[T3]], i32 2, i64 [[T2]])
  // CHECK-BOTHAUTH: [[T5:%.*]] = inttoptr i64 [[T4]] to ptr
  // CHECK-BOTHAUTH: [[T6:%.*]] = load volatile i8, ptr [[T5]], align 8
}

extern "C" const void *aArrayParameter(A aArray[]) {
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(aArray)), const void*>::value);
  // CHECK-NOAUTH: [[THIS_OBJ:%.*]] = load ptr, ptr %aArray.addr
  // CHECK-NOAUTH: %vtable = load ptr, ptr [[THIS_OBJ]]
  // CHECK-TYPEAUTH: [[THIS_OBJ:%.*]] = load ptr, ptr %aArray.addr
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr [[THIS_OBJ]]
  // CHECK-TYPEAUTH: [[VTABLEI:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI]], i32 2, i64 48388)
  // CHECK-ADDRESSAUTH: [[VTABLE_ADDR:%.*]] = load ptr, ptr %aArray.addr, align 8, !tbaa !2
  // CHECK-ADDRESSAUTH: %vtable = load ptr, ptr %0, align 8, !tbaa !7
  // CHECK-ADDRESSAUTH: [[VTABLE_ADDRI:%.*]] = ptrtoint ptr [[VTABLE_ADDR]] to i64
  // CHECK-ADDRESSAUTH: [[VTABLEI:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI]], i32 2, i64 [[VTABLE_ADDRI]])
  // CHECK-BOTHAUTH: [[VTABLE_ADDR:%.*]] = load ptr, ptr %aArray.addr, align 8, !tbaa !2
  // CHECK-BOTHAUTH: %vtable = load ptr, ptr [[VTABLE_ADDR]], align 8, !tbaa !7
  // CHECK-BOTHAUTH: [[VTABLE_ADDRI:%.*]] = ptrtoint ptr [[VTABLE_ADDR]] to i64
  // CHECK-BOTHAUTH: [[VTABLE_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[VTABLE_ADDRI]], i64 48388)
  // CHECK-BOTHAUTH: [[VTABLE_PTR:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLE_PTR]], i32 2, i64 [[VTABLE_DISC]])
  return __builtin_get_vtable_pointer(aArray);
}

extern "C" const void *aArrayLocal() {
  A array[] = { A() };
  static_assert(same_type<decltype(__builtin_get_vtable_pointer(array)), const void*>::value);
  // CHECK-NOAUTH: [[THIS_OBJ:%.*]] = getelementptr inbounds [1 x %"struct.test1::A"], ptr %array
  // CHECK-NOAUTH: %vtable = load ptr, ptr %arraydecay
  // CHECK-TYPEAUTH: %arraydecay = getelementptr inbounds [1 x %"struct.test1::A"]
  // CHECK-TYPEAUTH: %vtable = load ptr, ptr %arraydecay
  // CHECK-TYPEAUTH: [[VTABLEI:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-TYPEAUTH: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI]], i32 2, i64 48388)
  // CHECK-ADDRESSAUTH: %arraydecay = getelementptr inbounds [1 x %"struct.test1::A"], ptr %array, i64 0, i64 0
  // CHECK-ADDRESSAUTH: %vtable = load ptr, ptr %arraydecay, align 8, !tbaa !7
  // CHECK-ADDRESSAUTH: [[VTABLE_ADDRI:%.*]] = ptrtoint ptr %arraydecay to i64
  // CHECK-ADDRESSAUTH: [[VTABLEI:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-ADDRESSAUTH: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI]], i32 2, i64 [[VTABLE_ADDRI]])
  // CHECK-BOTHAUTH: %arraydecay = getelementptr inbounds [1 x %"struct.test1::A"], ptr %array, i64 0, i64 0
  // CHECK-BOTHAUTH: %vtable = load ptr, ptr %arraydecay, align 8, !tbaa !7
  // CHECK-BOTHAUTH: [[VTABLE_ADDRI:%.*]] = ptrtoint ptr %arraydecay to i64
  // CHECK-BOTHAUTH: [[VTABLE_DISC:%.*]] = call i64 @llvm.ptrauth.blend(i64 %0, i64 48388)
  // CHECK-BOTHAUTH: [[VTABLEI:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-BOTHAUTH: [[AUTHENTICATED:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[VTABLEI]], i32 2, i64 [[VTABLE_DISC]])
  return __builtin_get_vtable_pointer(array);
}

void test() {
  A aInstance;
  B bInstance;
  C cInstance;
  D dInstance;
  E eInstance;
  E eArray[] = { E() };
  a(&aInstance);
  a(&bInstance);
  a((B *)&cInstance);
  a(&dInstance);
  a((D *)&eInstance);
  a((B *)&eInstance);
  b(&bInstance);
  b(&cInstance);
  b(&eInstance);
  b_as_A(&bInstance);
  c(&cInstance);
  c_as_Z(&cInstance);
  c_as_B(&cInstance);
  d(&dInstance);
  d(&eInstance);
  d_as_A(&dInstance);
  d_as_A(&eInstance);
  e(&eInstance);
  e_as_B(&eInstance);
  e_as_D(&eInstance);
  (void)__builtin_get_vtable_pointer(eArray);
}
} // namespace test1
