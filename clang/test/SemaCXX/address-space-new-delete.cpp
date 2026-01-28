// RUN: %clang_cc1 -std=c++17 %s -verify

typedef int FOO __attribute__((opencl_local));

void test_new() {
  int *p = new FOO[1];
  // expected-error@-1 {{'new' cannot allocate objects of type 'int' in address space}}
}

void test_delete(FOO *p) {
  delete p;
  // expected-error@-1 {{cannot delete objects of type 'int' in address space}}
}
