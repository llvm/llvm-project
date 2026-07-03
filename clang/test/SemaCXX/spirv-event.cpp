// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++11 -triple spirv64 -Wno-unused-value %s

// __spirv_event_t is an opaque type: it cannot be initialized from, converted
// to, or cast from other types, nor used in arithmetic.
void foo() {
  int n = 100;
  __spirv_event_t v = 0; // expected-error {{cannot initialize a variable of type '__spirv_event_t' with an rvalue of type 'int'}}
  static_cast<__spirv_event_t>(n); // expected-error {{static_cast from 'int' to '__spirv_event_t' is not allowed}}
  reinterpret_cast<__spirv_event_t>(n); // expected-error {{reinterpret_cast from 'int' to '__spirv_event_t' is not allowed}}
  (void)(v + v); // expected-error {{invalid operands to binary expression ('__spirv_event_t' and '__spirv_event_t')}}
  int x(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__spirv_event_t'}}
  __spirv_event_t k;
  int *ip = (int *)k; // expected-error {{cannot cast from type '__spirv_event_t' to pointer type 'int *'}}
}

// __spirv_event_t can be used as a function parameter, a template argument, and
// a struct field.
template <class T> void bar(T);
void use(__spirv_event_t r) { bar(r); }
struct S { __spirv_event_t r; int a; };
