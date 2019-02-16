// RUN: %clangxx -frtti -fsanitize=null,vptr -g %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// REQUIRES: cxxabi
// UNSUPPORTED: windows-msvc

#include <string.h>

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
  char *c = new char[sizeof(Derived)];
  memset((void *)c, 0xFF, sizeof(Derived));
  Derived *list = (Derived *)c;

// CHECK: PR33221.cpp:[[@LINE+2]]:19: runtime error: member access within address {{.*}} which does not point to an object of type 'Base'
// CHECK-NEXT: invalid vptr
  int foo = list->i;
  return 0;
}
