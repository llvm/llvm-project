// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux -emit-llvm -fsanitize=null %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NULL --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows -emit-llvm -fsanitize=null %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NULL --check-prefix=MSABI
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-linux -emit-llvm -fsanitize=null,vptr %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-VPTR --check-prefix=ITANIUM
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows -emit-llvm -fsanitize=null,vptr %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-VPTR --check-prefix=MSABI
struct T {
  virtual ~T() {}
  virtual int v() { return 1; }
};

struct U : T {
  ~U();
  virtual int v() { return 2; }
};

U::~U() {}

// ITANIUM: define i32 @_Z5get_vP1T
// MSABI: define i32 @"\01?get_v
int get_v(T* t) {
  // First, we check that vtable is not loaded before a type check.
  // CHECK-NULL-NOT: load {{.*}} (%struct.T*{{.*}})**, {{.*}} (%struct.T*{{.*}})***
  // CHECK-NULL: [[UBSAN_CMP_RES:%[0-9]+]] = icmp ne %struct.T* %{{[_a-z0-9]+}}, null
  // CHECK-NULL-NEXT: br i1 [[UBSAN_CMP_RES]], label %{{.*}}, label %{{.*}}
  // CHECK-NULL: call void @__ubsan_handle_type_mismatch_v1_abort
  // Second, we check that vtable is actually loaded once the type check is done.
  // CHECK-NULL: load {{.*}} (%struct.T*{{.*}})**, {{.*}} (%struct.T*{{.*}})***
  return t->v();
}

// ITANIUM: define void @_Z9delete_itP1T
// MSABI: define void @"\01?delete_it
void delete_it(T *t) {
  // First, we check that vtable is not loaded before a type check.
  // CHECK-VPTR-NOT: load {{.*}} (%struct.T*{{.*}})**, {{.*}} (%struct.T*{{.*}})***
  // CHECK-VPTR: br i1 {{.*}} label %{{.*}}
  // CHECK-VPTR: call void @__ubsan_handle_dynamic_type_cache_miss_abort
  // Second, we check that vtable is actually loaded once the type check is done.
  // CHECK-VPTR: load {{.*}} (%struct.T*{{.*}})**, {{.*}} (%struct.T*{{.*}})***
  delete t;
}

// ITANIUM: define %struct.U* @_Z7dyncastP1T
// MSABI: define %struct.U* @"\01?dyncast
U* dyncast(T *t) {
  // First, we check that dynamic_cast is not called before a type check.
  // CHECK-VPTR-NOT: call i8* @__{{dynamic_cast|RTDynamicCast}}
  // CHECK-VPTR: br i1 {{.*}} label %{{.*}}
  // CHECK-VPTR: call void @__ubsan_handle_dynamic_type_cache_miss_abort
  // Second, we check that dynamic_cast is actually called once the type check is done.
  // CHECK-VPTR: call i8* @__{{dynamic_cast|RTDynamicCast}}
  return dynamic_cast<U*>(t);
}
