// RUN: %clang_cc1 -std=c++17 -triple spirv64-amd-amdhsa -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++17 -triple spirv64-amd-amdhsa -fcxx-exceptions -fexceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s

// The destructor pointer passed to __cxa_throw uses the program address space.

struct X {
  ~X();
};

struct Guard {
  ~Guard();
};

void throw_int() { throw 1; }

void throw_object() { throw X{}; }

void throw_with_cleanup() {
  Guard guard;
  throw 2;
}

void throw_object_with_cleanup() {
  Guard guard;
  throw X{};
}

// CHECK-LABEL: define{{.*}} void @_Z9throw_intv()
// CHECK: call {{.*}}void @__cxa_throw(ptr{{.*}} %{{.*}}, ptr {{.*}}@_ZTIi, ptr addrspace(4) null)
// CHECK-LABEL: define{{.*}} void @_Z12throw_objectv()
// CHECK: call {{.*}}void @__cxa_throw(ptr{{.*}} %{{.*}}, ptr {{.*}}@_ZTI1X, ptr addrspace(4) @_ZN1XD1Ev)
// CHECK-LABEL: define{{.*}} void @_Z18throw_with_cleanupv()
// CHECK: invoke {{.*}}void @__cxa_throw(ptr{{.*}} %{{.*}}, ptr {{.*}}@_ZTIi, ptr addrspace(4) null)
// CHECK-LABEL: define{{.*}} void @_Z25throw_object_with_cleanupv()
// CHECK: invoke {{.*}}void @__cxa_throw(ptr{{.*}} %{{.*}}, ptr {{.*}}@_ZTI1X, ptr addrspace(4) @_ZN1XD1Ev)
