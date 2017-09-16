// RUN: %clangxx -frtti -fsanitize=null,vptr -fno-sanitize-recover=vptr -g %s -O3 -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// REQUIRES: cxxabi
// UNSUPPORTED: win32

struct S { virtual int f() { return 0; } };
struct T : virtual S {};

struct Foo { virtual int f() { return 0; } };

int main(int argc, char **argv) {
  Foo foo;
  T *t = (T*)&foo;
  S *s = t;
  // CHECK: vptr-virtual-base.cpp:[[@LINE-1]]:10: runtime error: cast to virtual base of address [[PTR:0x[0-9a-f]*]] which does not point to an object of type 'T'
  // CHECK-NEXT: [[PTR]]: note: object is of type 'Foo'
  return s->f();
}
