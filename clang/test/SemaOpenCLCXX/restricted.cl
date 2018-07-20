// RUN: %clang_cc1 %s -triple spir-unknown-unknown -cl-std=c++ -pedantic -verify -fsyntax-only

// This test checks that various C/C++/OpenCL C constructs are not available in
// OpenCL C++, according to OpenCL C++ 1.0 Specification Section 2.9.

// Test that typeid is not available in OpenCL C++.
namespace std {
  // Provide a dummy std::type_info so that we can use typeid.
  class type_info {
    int a;
  };
}
__constant std::type_info int_ti = typeid(int);
// expected-error@-1 {{'typeid' is not supported in OpenCL C++}}

// Test that dynamic_cast is not available in OpenCL C++.
class A {
public:
  int a;
};

class B : public A {
  int b;
};

B *test_dynamic_cast(B *p) {
  return dynamic_cast<B *>(p);
  // expected-error@-1 {{'dynamic_cast' is not supported in OpenCL C++}}
}

// Test storage class qualifiers.
__constant _Thread_local int a = 1;
// expected-error@-1 {{OpenCL C++ version 1.0 does not support the '_Thread_local' storage class specifier}}
__constant __thread int b = 2;
// expected-error@-1 {{OpenCL C++ version 1.0 does not support the '__thread' storage class specifier}}
kernel void test_storage_classes() {
  register int x;
  // expected-error@-1 {{OpenCL C++ version 1.0 does not support the 'register' storage class specifier}}
  thread_local int y;
  // expected-error@-1 {{OpenCL C++ version 1.0 does not support the 'thread_local' storage class specifier}}
}

// Test that access qualifiers are reserved keywords.
kernel void test_access_qualifiers() {
  int read_only;
  // expected-error@-1 {{'read_only' is a reserved keyword in OpenCL C++}}
  // expected-warning@-2 {{declaration does not declare anything}}
  int __read_only;
  // expected-error@-1 {{'__read_only' is a reserved keyword in OpenCL C++}}
  // expected-warning@-2 {{declaration does not declare anything}}
  int write_only;
  // expected-error@-1 {{'write_only' is a reserved keyword in OpenCL C++}}
  // expected-warning@-2 {{declaration does not declare anything}}
  int __write_only;
  // expected-error@-1 {{'__write_only' is a reserved keyword in OpenCL C++}}
  // expected-warning@-2 {{declaration does not declare anything}}
  int read_write;
  // expected-error@-1 {{'read_write' is a reserved keyword in OpenCL C++}}
  // expected-warning@-2 {{declaration does not declare anything}}
  int __read_write;
  // expected-error@-1 {{'__read_write' is a reserved keyword in OpenCL C++}}
  // expected-warning@-2 {{declaration does not declare anything}}
}
