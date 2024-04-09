// RUN: %clang_cc1 %s -triple spirv64-unknown-unknown -fsycl-is-device -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s

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

// CHECK: declare{{.*}} void @__cxa_throw(ptr addrspace(4), ptr addrspace(1), ptr addrspace(4))
