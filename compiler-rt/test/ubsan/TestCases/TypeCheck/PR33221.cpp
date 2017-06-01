// RUN: %clangxx -frtti -fsanitize=undefined -g %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: cxxabi

class Base {
public:
  int i;
  virtual void print() {}
};

class Derived : public Base {
public:
  void print() {}
};

int main() {
  Derived *list = (Derived *)new char[sizeof(Derived)];

// CHECK: PR33221.cpp:[[@LINE+2]]:19: runtime error: member access within address {{.*}} which does not point to an object of type 'Base'
// CHECK-NEXT: object has invalid vptr
  int foo = list->i;
  return 0;
}
