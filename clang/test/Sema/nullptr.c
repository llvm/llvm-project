// RUN: %clang_cc1 -fsyntax-only -verify -std=c2x -ffreestanding -Wno-null-conversion -Wno-tautological-compare %s
#include <stdint.h>

typedef typeof(nullptr) nullptr_t;

struct A {};

__attribute__((overloadable)) int o1(char*);
__attribute__((overloadable)) void o1(uintptr_t);

nullptr_t f(nullptr_t null)
{
  // Implicit conversions.
  null = nullptr;
  void *p = nullptr;
  p = null;
  int *pi = nullptr;
  pi = null;
  null = 0; // expected-error {{assigning to 'nullptr_t' from incompatible type 'int'}}
  bool b = nullptr; // expected-error {{initializing 'bool' with an expression of incompatible type 'nullptr_t'}}

  // Can't convert nullptr to integral implicitly.
  uintptr_t i = nullptr; // expected-error-re {{initializing 'uintptr_t' (aka '{{.*}}') with an expression of incompatible type 'nullptr_t'}}

  // Operators
  (void)(null == nullptr);
  (void)(null <= nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(null == 0);
  (void)(null == (void*)0);
  (void)((void*)0 == nullptr);
  (void)(null <= 0); // expected-error {{invalid operands to binary expression}}
  (void)(null <= (void*)0); // expected-error {{invalid operands to binary expression}}
  (void)((void*)0 <= nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(0 == nullptr);
  (void)(nullptr == 0);
  (void)(nullptr <= 0); // expected-error {{invalid operands to binary expression}}
  (void)(0 <= nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(1 > nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(1 != nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(1 + nullptr); // expected-error {{invalid operands to binary expression}}
  (void)(0 ? nullptr : 0); // expected-error {{non-pointer operand type 'int' incompatible with nullptr}}
  (void)(0 ? nullptr : (void*)0);
  (void)(0 ? nullptr : (struct A){}); // expected-error {{non-pointer operand type 'struct A' incompatible with nullptr}}
  (void)(0 ? (struct A){} : nullptr); // expected-error {{non-pointer operand type 'struct A' incompatible with nullptr}}

  // Overloading
  int t = o1(nullptr);
  t = o1(null);

  // nullptr is an rvalue, null is an lvalue
  (void)&nullptr; // expected-error {{cannot take the address of an rvalue of type 'nullptr_t'}}
  nullptr_t *pn = &null;

  int *ip = *pn;
  if (*pn) { }
}

__attribute__((overloadable)) void *g(void*);
__attribute__((overloadable)) bool g(bool);

// Test that we prefer g(void*) over g(bool).
static_assert(__builtin_types_compatible_p(typeof(g(nullptr)), void *), "");

void sent(int, ...) __attribute__((sentinel));

void g() {
  // nullptr can be used as the sentinel value.
  sent(10, nullptr);
}

void printf(const char*, ...) __attribute__((format(printf, 1, 2)));

void h() {
  // Don't warn when using nullptr with %p.
  printf("%p", nullptr);
}

static_assert(sizeof(nullptr_t) == sizeof(void*), "");

static_assert(!(nullptr < nullptr), ""); // expected-error {{invalid operands to binary expression}}
static_assert(!(nullptr > nullptr), ""); // expected-error {{invalid operands to binary expression}}
static_assert(  nullptr <= nullptr, ""); // expected-error {{invalid operands to binary expression}}
static_assert(  nullptr >= nullptr, ""); // expected-error {{invalid operands to binary expression}}
static_assert(  nullptr == nullptr, "");
static_assert(!(nullptr != nullptr), "");

static_assert(!(0 < nullptr), ""); // expected-error {{invalid operands to binary expression}}
static_assert(!(0 > nullptr), ""); // expected-error {{invalid operands to binary expression}}
static_assert(  0 <= nullptr, ""); // expected-error {{invalid operands to binary expression}}
static_assert(  0 >= nullptr, ""); // expected-error {{invalid operands to binary expression}}
static_assert(  0 == nullptr, "");
static_assert(!(0 != nullptr), "");

static_assert(!(nullptr < 0), ""); // expected-error {{invalid operands to binary expression}}
static_assert(!(nullptr > 0), ""); // expected-error {{invalid operands to binary expression}}
static_assert(  nullptr <= 0, ""); // expected-error {{invalid operands to binary expression}}
static_assert(  nullptr >= 0, ""); // expected-error {{invalid operands to binary expression}}
static_assert(  nullptr == 0, "");
static_assert(!(nullptr != 0), "");

__attribute__((overloadable)) int f1(int*);
__attribute__((overloadable)) float f1(bool);

void test_f1() {
  int ir = (f1)(nullptr);
}

