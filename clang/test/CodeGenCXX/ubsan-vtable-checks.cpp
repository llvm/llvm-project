// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux -emit-llvm -fsanitize=null %s -o - | FileCheck %s --check-prefix=CHECK-NULL --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows -emit-llvm -fsanitize=null %s -o - | FileCheck %s --check-prefix=CHECK-NULL --check-prefix=MSABI
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux -emit-llvm -fsanitize=null,vptr %s -o - | FileCheck %s --check-prefix=CHECK-VPTR --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows -emit-llvm -fsanitize=null,vptr %s -o - | FileCheck %s --check-prefix=CHECK-VPTR --check-prefix=MSABI  --check-prefix=CHECK-VPTR-MS
// RUN: %clang_cc1 -std=c++11 -triple arm64e-ios-13 -emit-llvm -fptrauth-intrinsics -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-address-discrimination -fsanitize=null,vptr %s -o - | FileCheck %s --check-prefix=CHECK-VPTR --check-prefix=ITANIUM --check-prefix=CHECK-PTRAUTH
// RUN: %clang_cc1 -std=c++11 -triple aarch64-unknown-linux -emit-llvm -fptrauth-intrinsics -fptrauth-calls -fptrauth-vtable-pointer-type-discrimination -fptrauth-vtable-pointer-address-discrimination -fsanitize=null,vptr %s -o - | FileCheck %s --check-prefix=CHECK-VPTR --check-prefix=ITANIUM --check-prefix=CHECK-PTRAUTH

struct T {
  virtual ~T() {}
  virtual int v() { return 1; }
};

struct U : T {
  ~U();
  virtual int v() { return 2; }
};

U::~U() {}

// CHECK-VPTR-MS: @__ubsan_vptr_type_cache = external dso_local

// ITANIUM: define{{.*}} i32 @_Z5get_vP1T
// MSABI: define dso_local noundef i32 @"?get_v
int get_v(T* t) {
  // First, we check that vtable is not loaded before a type check.
  // CHECK-NULL-NOT: load {{.*}} (ptr{{.*}})**, {{.*}} (ptr{{.*}})***
  // CHECK-NULL: [[UBSAN_CMP_RES:%[0-9]+]] = icmp ne ptr %{{[_a-z0-9]+}}, null
  // CHECK-NULL-NEXT: br i1 [[UBSAN_CMP_RES]], label %{{.*}}, label %{{.*}}
  // CHECK-NULL: call void @__ubsan_handle_type_mismatch_v1_abort
  // Second, we check that vtable is actually loaded once the type check is done.
  // CHECK-NULL: load ptr, ptr {{.*}}

  // CHECK-PTRAUTH: [[CAST_VTABLE:%.*]] = ptrtoint ptr %vtable to i64
  // CHECK-PTRAUTH: [[STRIPPED_VTABLE:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[CAST_VTABLE]], i32 0), !nosanitize
  // CHECK-PTRAUTH: [[STRIPPED_PTR:%.*]] = inttoptr i64 [[STRIPPED_VTABLE]] to ptr
  // CHECK-PTRAUTH: [[STRIPPED_INT:%.*]] = ptrtoint ptr [[STRIPPED_PTR]] to i64
  // Make sure authed vtable pointer feeds into hashing
  // CHECK-PTRAUTH: {{%.*}} = mul i64 [[STRIPPED_INT]], {{.*}}

  // Verify that we authenticate for the actual vcall
  // CHECK-PTRAUTH: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 17113)
  // CHECK-PTRAUTH: [[CAST_VTABLE:%.*]] = ptrtoint ptr %vtable2 to i64
  // CHECK-PTRAUTH: [[AUTHED_INT:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_VTABLE]], i32 2, i64 [[BLENDED]])
  // CHECK-PTRAUTH: [[AUTHED_PTR:%.*]] = inttoptr i64 [[AUTHED_INT]] to ptr
  // CHECK-PTRAUTH: {{%.*}} = getelementptr inbounds ptr, ptr [[AUTHED_PTR]], i64 2
  return t->v();
}

// ITANIUM: define{{.*}} void @_Z9delete_itP1T
// MSABI: define dso_local void @"?delete_it
void delete_it(T *t) {
  // CHECK-VPTR-NOT: load {{.*}} (ptr{{.*}})**, {{.*}} (ptr{{.*}})***
  // CHECK-VPTR: br i1 {{.*}} label %{{.*}}
  // CHECK-VPTR: call void @__ubsan_handle_type_mismatch_v1_abort
  // Second, we check that vtable is actually loaded once the type check is done.
  // CHECK-VPTR: load ptr, ptr {{.*}}

  // First, we check that vtable is not loaded before a type check.
  // CHECK-PTRAUTH: ptrtoint ptr {{%.*}} to i64
  // CHECK-PTRAUTH: [[CAST_VTABLE:%.*]] = ptrtoint ptr [[VTABLE:%.*]] to i64
  // CHECK-PTRAUTH: [[STRIPPED_VTABLE:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[CAST_VTABLE]], i32 0)
  // CHECK-PTRAUTH: [[STRIPPED_PTR:%.*]] = inttoptr i64 [[STRIPPED_VTABLE]] to ptr
  // CHECK-PTRAUTH: [[STRIPPED_INT:%.*]] = ptrtoint ptr [[STRIPPED_PTR]] to i64
  // CHECK-PTRAUTH: {{%.*}} = mul i64 [[STRIPPED_INT]], {{.*}}
  // CHECK-PTRAUTH: call void @__ubsan_handle_dynamic_type_cache_miss_abort(
  // Second, we check that vtable is actually loaded once the type check is done.
  // ptrauth for the virtual function load
  // CHECK-PTRAUTH: [[VTABLE2:%.*]] = load ptr, ptr {{.*}}
  // CHECK-PTRAUTH: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 %{{.*}}, i64 17113)
  // CHECK-PTRAUTH: [[CAST_VTABLE:%.*]] = ptrtoint ptr [[VTABLE2]] to i64
  // CHECK-PTRAUTH: [[AUTHED_INT:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_VTABLE]], i32 2, i64 [[BLENDED]])
  // CHECK-PTRAUTH: [[AUTHED_PTR:%.*]] = inttoptr i64 [[AUTHED_INT]] to ptr
  // CHECK-PTRAUTH: getelementptr inbounds ptr, ptr
  // CHECK-PTRAUTH {{%.*}} = getelementptr inbounds ptr, ptr [[AUTHED_PTR]], i64 1
  delete t;
}

// ITANIUM: define{{.*}} ptr @_Z7dyncastP1T
// MSABI: define dso_local noundef ptr @"?dyncast
U* dyncast(T *t) {
  // First, we check that dynamic_cast is not called before a type check.
  // CHECK-VPTR-NOT: call ptr @__{{dynamic_cast|RTDynamicCast}}
  // CHECK-VPTR: br i1 {{.*}} label %{{.*}}
  // CHECK-PTRAUTH: [[V0:%.*]] = ptrtoint ptr {{%.*}} to i64
  // CHECK-PTRAUTH: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 [[V0]], i64 17113)
  // CHECK-PTRAUTH: [[CAST_VTABLE:%.*]] = ptrtoint ptr {{%.*}} to i64
  // CHECK-PTRAUTH: [[STRIPPED_VTABLE:%.*]] = call i64 @llvm.ptrauth.strip(i64 [[CAST_VTABLE]], i32 0)
  // CHECK-PTRAUTH: [[STRIPPED_PTR:%.*]] = inttoptr i64 [[STRIPPED_VTABLE]] to ptr
  // CHECK-PTRAUTH: [[STRIPPED_INT:%.*]] = ptrtoint ptr [[STRIPPED_PTR]] to i64
  // CHECK-PTRAUTH: {{%.*}} = mul i64 [[STRIPPED_INT]], {{.*}}
  // CHECK-VPTR: call void @__ubsan_handle_dynamic_type_cache_miss_abort
  // CHECK-PTRAUTH: [[BLENDED:%.*]] = call i64 @llvm.ptrauth.blend(i64 {{%.*}}, i64 17113)
  // CHECK-PTRAUTH: [[CAST_VTABLE:%.*]] = ptrtoint ptr %vtable1 to i64
  // CHECK-PTRAUTH: [[AUTHED_INT:%.*]] = call i64 @llvm.ptrauth.auth(i64 [[CAST_VTABLE]], i32 2, i64 [[BLENDED]])
  // CHECK-PTRAUTH: [[AUTHED_PTR:%.*]] = inttoptr i64 [[AUTHED_INT]] to ptr
  // CHECK-PTRAUTH: {{%.*}} = load volatile i8, ptr [[AUTHED_PTR]], align 8
  // Second, we check that dynamic_cast is actually called once the type check is done.
  // CHECK-VPTR: call ptr @__{{dynamic_cast|RTDynamicCast}}
  return dynamic_cast<U*>(t);
}
