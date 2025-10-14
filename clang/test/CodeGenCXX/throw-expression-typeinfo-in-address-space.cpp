// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple spirv64-amd-amdhsa -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s --check-prefix=WITH-NONZERO-DEFAULT-AS

struct X {
  ~X();
};

struct Error {
  Error(const X&) noexcept;
};

void f() {
  try {
    throw Error(X());
  } catch (...) { }
}

// CHECK: declare void @__cxa_throw(ptr, ptr addrspace(1), ptr)
// WITH-NONZERO-DEFAULT-AS: declare{{.*}} void @__cxa_throw(ptr addrspace(4), ptr addrspace(1), ptr addrspace(4))
