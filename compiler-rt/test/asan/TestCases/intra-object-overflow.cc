// RUN: %clangxx_asan -O0 -fsanitize-address-field-padding=1  %s -o %t
// RUN: not %run %t 11 2>&1 | FileCheck %s
// RUN: %run %t 10
//
// FIXME: fix 32-bits.
// REQUIRES: asan-64-bits, shadow-scale-3
// FIXME: Implement ASan intra-object padding in Clang's MS record layout
// UNSUPPORTED: windows-msvc
#include <stdio.h>
#include <stdlib.h>
class Foo {
 public:
  Foo() : pre1(1), pre2(2), post1(3), post2(4) {
  }
  virtual ~Foo() {
  }
  void set(int i, int val) { a[i] = val; }
// CHECK: ERROR: AddressSanitizer: intra-object-overflow
// CHECK: #0 {{.*}}Foo::set{{.*}}intra-object-overflow.cc:[[@LINE-2]]
 private:
  int pre1, pre2;
  int a[11];
  int post1, post2;
};

int main(int argc, char **argv) {
  int idx = argc == 2 ? atoi(argv[1]) : 0;
  Foo *foo = new Foo;
  foo->set(idx, 42);
// CHECK: #1 {{.*}}main{{.*}}intra-object-overflow.cc:[[@LINE-1]]
// CHECK: is located 84 bytes inside of 128-byte region
  delete foo;
}
