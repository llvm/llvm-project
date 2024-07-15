// RUN: %clang_cc1 -verify -ffreestanding -Wno-unused -std=c2x %s

/* WG14 N3042: full
 * Introduce the nullptr constant
 */

#include <stddef.h>

// FIXME: The paper calls for a feature testing macro to be added to stddef.h
// which we do not implement. This should be addressed after WG14 has processed
// national body comments for C2x as we've asked for the feature test macros to
// be removed.
#ifndef __STDC_VERSION_STDDEF_H__
#error "no version macro for stddef.h"
#endif
// expected-error@-2 {{"no version macro for stddef.h"}}

void questionable_behaviors() {
  nullptr_t val;

  // This code is intended to be rejected by C and is accepted by C++. We filed
  // an NB comment asking for this to be changed, but WG14 declined.
  (void)(1 ? val : 0);     // expected-error {{non-pointer operand type 'int' incompatible with nullptr}}
  (void)(1 ? nullptr : 0); // expected-error {{non-pointer operand type 'int' incompatible with nullptr}}

  // This code is intended to be accepted by C and is rejected by C++. We filed
  // an NB comment asking for this to be changed, but WG14 declined.
  _Bool another = val;    // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  another = val;          // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  _Bool again = nullptr;  // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  again = nullptr;        // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
}

void test() {
  // Can we declare the type?
  nullptr_t null_val;

  // Can we use the keyword?
  int *typed_ptr = nullptr;
  typed_ptr = nullptr;

  // Can we use the keyword with the type?
  null_val = nullptr;
  // Even initialize with it?
  nullptr_t ignore = nullptr;

  // Can we assign an object of the type to another object of the same type?
  null_val = null_val;

  // Can we assign nullptr_t objects to pointer objects?
  typed_ptr = null_val;

  // Can we take the address of an object of type nullptr_t?
  &null_val;

  // How about the null pointer named constant?
  &nullptr; // expected-error {{cannot take the address of an rvalue of type 'nullptr_t'}}

  // Assignment from a null pointer constant to a nullptr_t is valid.
  null_val = 0;
  null_val = (void *)0;

  // Assignment from a nullptr_t to a pointer is also valid.
  typed_ptr = null_val;
  void *other_ptr = null_val;

  // Can it be used in all the places a scalar can be used?
  if (null_val) {}     // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  if (!null_val) {}    // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  for (;null_val;) {}  // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  while (nullptr) {}   // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  null_val && nullptr; // expected-warning {{implicit conversion of nullptr constant to 'bool'}} \
                          expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  nullptr || null_val; // expected-warning {{implicit conversion of nullptr constant to 'bool'}} \
                          expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  null_val ? 0 : 1;    // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  sizeof(null_val);
  alignas(nullptr_t) int aligned;

  // Cast expressions have special handling for nullptr_t despite allowing
  // casts of scalar types.
  (nullptr_t)12;        // expected-error {{cannot cast an object of type 'int' to 'nullptr_t'}}
  (float)null_val;      // expected-error {{cannot cast an object of type 'nullptr_t' to 'float'}}
  (float)nullptr;       // expected-error {{cannot cast an object of type 'nullptr_t' to 'float'}}
  (nullptr_t)0;         // expected-error {{cannot cast an object of type 'int' to 'nullptr_t'}}
  (nullptr_t)(void *)0; // expected-error {{cannot cast an object of type 'void *' to 'nullptr_t'}}
  (nullptr_t)(int *)12; // expected-error {{cannot cast an object of type 'int *' to 'nullptr_t'}}

  (void)null_val;     // ok
  (void)nullptr;      // ok
  (bool)null_val;     // ok
  (bool)nullptr;      // ok
  (int *)null_val;    // ok
  (int *)nullptr;     // ok
  (nullptr_t)nullptr; // ok

  // Can it be converted to bool with the result false (this relies on Clang
  // accepting additional kinds of constant expressions where an ICE is
  // required)?
  static_assert(!nullptr);  // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  static_assert(!null_val); // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  static_assert(nullptr);   // expected-error {{static assertion failed due to requirement 'nullptr'}} \
                               expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  static_assert(null_val);  // expected-error {{static assertion failed due to requirement 'null_val'}} \
                               expected-warning {{implicit conversion of nullptr constant to 'bool'}}

  // Do equality operators work as expected with it?
  static_assert(nullptr == nullptr);
  static_assert(null_val == null_val);
  static_assert(nullptr != (int*)1);
  static_assert(null_val != (int*)1);
  static_assert(nullptr == null_val);
  static_assert(nullptr == 0);
  static_assert(null_val == (void *)0);

  // None of the relational operators should succeed.
  (void)(null_val <= 0);            // expected-error {{invalid operands to binary expression ('nullptr_t' and 'int')}}
  (void)(null_val >= (void *)0);    // expected-error {{invalid operands to binary expression ('nullptr_t' and 'void *')}}
  (void)(!(null_val < (void *)0));  // expected-error {{invalid operands to binary expression ('nullptr_t' and 'void *')}}
  (void)(!(null_val > 0));          // expected-error {{invalid operands to binary expression ('nullptr_t' and 'int')}}
  (void)(nullptr <= 0);             // expected-error {{invalid operands to binary expression ('nullptr_t' and 'int')}}
  (void)(nullptr >= (void *)0);     // expected-error {{invalid operands to binary expression ('nullptr_t' and 'void *')}}
  (void)(!(nullptr < (void *)0));   // expected-error {{invalid operands to binary expression ('nullptr_t' and 'void *')}}
  (void)(!(nullptr > 0));           // expected-error {{invalid operands to binary expression ('nullptr_t' and 'int')}}
  (void)(null_val <= null_val);     // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(null_val >= null_val);     // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(!(null_val < null_val));   // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(!(null_val > null_val));   // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(null_val <= nullptr);      // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(null_val >= nullptr);      // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(!(null_val < nullptr));    // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(!(null_val > nullptr));    // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(nullptr <= nullptr);       // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(nullptr >= nullptr);       // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(!(nullptr < nullptr));     // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}
  (void)(!(nullptr > nullptr));     // expected-error {{invalid operands to binary expression ('nullptr_t' and 'nullptr_t')}}

  // Do we pick the correct common type for conditional operators?
  _Generic(1 ? nullptr : nullptr, nullptr_t : 0);
  _Generic(1 ? null_val : null_val, nullptr_t : 0);
  _Generic(1 ? typed_ptr : null_val, typeof(typed_ptr) : 0);
  _Generic(1 ? null_val : typed_ptr, typeof(typed_ptr) : 0);
  _Generic(1 ? nullptr : typed_ptr, typeof(typed_ptr) : 0);
  _Generic(1 ? typed_ptr : nullptr, typeof(typed_ptr) : 0);

  // Same for GNU conditional operators?
  _Generic(nullptr ?: nullptr, nullptr_t : 0);            // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  _Generic(null_val ?: null_val, nullptr_t : 0);          // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  _Generic(typed_ptr ?: null_val, typeof(typed_ptr) : 0);
  _Generic(null_val ?: typed_ptr, typeof(typed_ptr) : 0); // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  _Generic(nullptr ?: typed_ptr, typeof(typed_ptr) : 0);  // expected-warning {{implicit conversion of nullptr constant to 'bool'}}
  _Generic(typed_ptr ?: nullptr, typeof(typed_ptr) : 0);

  // Do we correctly issue type incompatibility diagnostics?
  int i = nullptr;   // expected-error {{initializing 'int' with an expression of incompatible type 'nullptr_t'}}
  float f = nullptr; // expected-error {{initializing 'float' with an expression of incompatible type 'nullptr_t'}}
  i = null_val;      // expected-error {{assigning to 'int' from incompatible type 'nullptr_t'}}
  f = null_val;      // expected-error {{assigning to 'float' from incompatible type 'nullptr_t'}}
  null_val = i;      // expected-error {{assigning to 'nullptr_t' from incompatible type 'int'}}
  null_val = f;      // expected-error {{assigning to 'nullptr_t' from incompatible type 'float'}}
}

// Can we use it as a function parameter?
void null_param(nullptr_t);

void other_test() {
  // Can we call the function properly?
  null_param(nullptr);

  // We can pass any kind of null pointer constant.
  null_param((void *)0);
  null_param(0);
}


void printf(const char*, ...) __attribute__((format(printf, 1, 2)));
void format_specifiers() {
  // Don't warn when using nullptr with %p.
  printf("%p", nullptr);
}

// Ensure that conversion from a null pointer constant to nullptr_t is
// valid in a constant expression.
static_assert((nullptr_t){} == 0);
