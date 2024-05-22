// RUN: %clang_cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s

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
