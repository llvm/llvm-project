// RUN: %clang_cc1 -std=c++17 %s -verify -Wno-unknown-attributes

typedef int LocalInt __attribute__((opencl_local));

void test_new() {
  int *p = new LocalInt[1]; // expected-error {{cannot allocate objects of type 'int' in address space}}
}

void test_delete(LocalInt *p) {
  delete p; // expected-error {{cannot delete objects of type 'int' in address space}}
}
