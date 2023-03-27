// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s
//
// This test makes sure ASan symbolizes stack traces the way they are typically
// symbolized on Windows.
#include <malloc.h>

namespace foo {
// A template function in a namespace.
template<int x>
void bar(char *p) {
  *p = x;
}

// A regular function in a namespace.
void spam(char *p) {
  bar<42>(p);
}
}

// A multi-argument template with a bool template parameter.
template<typename T, bool U>
void baz(T t) {
  if (U)
    foo::spam(t);
}

template<typename T>
struct A {
  A(T v) { v_ = v; }
  ~A();
  char *v_;
};

// A destructor of a template class.
template<>
A<char*>::~A() {
  baz<char*, true>(v_);
}

int main() {
  char *buffer = (char*)malloc(42);
  free(buffer);
  A<char*> a(buffer);
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: foo::bar<42>{{.*}}demangled_names.cpp
// CHECK: foo::spam{{.*}}demangled_names.cpp
// CHECK: baz<char *,{{ *}}1>{{.*}}demangled_names.cpp
// CHECK: A<char *>::~A<char *>{{.*}}demangled_names.cpp
}
