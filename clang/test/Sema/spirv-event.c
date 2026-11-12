// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64 %s

__spirv_event_t g;
__spirv_event_t g_init = 0; // expected-error {{initializing '__spirv_event_t' with an expression of incompatible type 'int'}}

void foo(void) {
  __spirv_event_t v = 0; // expected-error {{initializing '__spirv_event_t' with an expression of incompatible type 'int'}}
  (void)(v + v); // expected-error {{invalid operands to binary expression ('__spirv_event_t' and '__spirv_event_t')}}
  int x = v; // expected-error {{initializing 'int' with an expression of incompatible type '__spirv_event_t'}}
  __spirv_event_t k;
  int *ip = (int *)k; // expected-error {{operand of type '__spirv_event_t' where arithmetic or pointer type is required}}
  (int)v; // expected-error {{operand of type '__spirv_event_t' where arithmetic or pointer type is required}}
  __spirv_event_t copy = k;
  (void)v; // Ok
}

void use(__spirv_event_t r);
__spirv_event_t make(void);
struct S { __spirv_event_t r; int a; };


