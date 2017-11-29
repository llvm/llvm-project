// RUN: %clang_cc1 -triple x86_64-linux-unknown -emit-llvm %s -o - | FileCheck %s

#pragma GCC visibility push(hidden)

struct Base {
  virtual ~Base() = default;
  virtual void* Alloc() = 0;
};

class Child : public Base {
public:
  Child() = default;
  void* Alloc();
};

void test() {
  Child x;
}

// CHECK: @_ZTV5Child = external hidden unnamed_addr constant
