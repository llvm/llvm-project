// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -O1 -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsanitize=memory -O2 -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

template <class T>
class Vector {
public:
  int size;
  ~Vector() {}
};

// Virtual function table for the derived class only contains
// its own destructors, with no aliasing to base class dtors.
struct Base {
  Vector<int> v;
  int x;
  Base() { x = 5; }
  virtual ~Base() {}
};

struct Derived : public Base {
  int z;
  Derived() { z = 10; }
  ~Derived() {}
};

Derived d;

// Definition of virtual function table
// CHECK: @_ZTV7Derived = {{.*}}@_ZN7DerivedD1Ev{{.*}}@_ZN7DerivedD0Ev
