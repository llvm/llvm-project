// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -std=c++11 | FileCheck %s
// PR13424

namespace Test1 {
struct X {
  virtual ~X(); // Key function.
  virtual void f(); // Not a key function.
};

X::~X() = default;

// Verify that the vtable is emitted.
// CHECK: @_ZTVN5Test11XE = unnamed_addr constant
}

namespace Test2 {
struct X {
  virtual ~X() = default; // Not a key function.
  virtual void f(); // Key function.
};

void X::f() {}

// Verify that the vtable is emitted.
// CHECK: @_ZTVN5Test21XE = unnamed_addr constant
}

namespace Test3 {
struct X {
  virtual ~X() = delete; // Not a key function.
  virtual void f(); // Key function.
};

void X::f() {}

// Verify that the vtable is emitted.
// CHECK: @_ZTVN5Test31XE = unnamed_addr constant
}
