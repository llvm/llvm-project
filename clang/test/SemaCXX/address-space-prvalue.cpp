// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify

struct S {
  S(int);
  void f();
  void cf() const;
};
struct Agg {
  int a;
};

using as1_S = __attribute__((address_space(1))) S;
using const_as1_S = const __attribute__((address_space(1))) S;
using as1_Agg = __attribute__((address_space(1))) Agg;
using as1_int = __attribute__((address_space(1))) int;

extern S s;
extern as1_S gs;

// Temporaries are created in the default address space, so a temporary whose
// type is qualified with a different address space is rejected.
void temporaries() {
  (void)as1_S{1};   // expected-error {{temporary of type 'as1_S' (aka '__attribute__((address_space(1))) S') cannot be created in address space '__attribute__((address_space(1)))'}}
  (void)as1_Agg{};  // expected-error {{temporary of type 'as1_Agg' (aka '__attribute__((address_space(1))) Agg') cannot be created in address space '__attribute__((address_space(1)))'}}
  (void)as1_Agg();  // expected-error {{temporary of type 'as1_Agg' (aka '__attribute__((address_space(1))) Agg') cannot be created in address space '__attribute__((address_space(1)))'}}

  // A cast to a class type initializes its result as a temporary of that type,
  // so it is rejected in the same way.
  (void)static_cast<as1_S>(s); // expected-error {{static_cast from 'S' to 'as1_S' (aka '__attribute__((address_space(1))) S') is not allowed}}
  (void)(as1_S)s;              // expected-error {{C-style cast from 'S' to 'as1_S' (aka '__attribute__((address_space(1))) S') is not allowed}}
  (void)as1_S(s);              // expected-error {{functional-style cast from 'S' to 'as1_S' (aka '__attribute__((address_space(1))) S') is not allowed}}
  (void)as1_Agg(1);            // expected-error {{functional-style cast from 'int' to 'as1_Agg' (aka '__attribute__((address_space(1))) Agg') is not allowed}}

  // Non-class prvalues drop their qualifiers.
  (void)as1_int{1};
  (void)as1_int(1);
  (void)static_cast<as1_int>(1);
}

// Traits should not emit diagnostics.
static_assert(!__is_constructible(as1_S, int));
static_assert(!__is_constructible(as1_Agg));
static_assert(!__is_constructible(as1_Agg, int));
static_assert(__is_constructible(as1_int, int));

// A variable is not a temporary and is not affected.
as1_Agg global_agg = {1};

using int_array = const __attribute__((address_space(1))) int[1];
using S_array = const __attribute__((address_space(1))) S[1];
void arrays() {
  (void)int_array{1}; // expected-error {{temporary of type 'int_array' (aka 'const __attribute__((address_space(1))) int[1]') cannot be created in address space '__attribute__((address_space(1)))'}}
  (void)S_array{1};   // expected-error {{temporary of type 'S_array' (aka 'const __attribute__((address_space(1))) S[1]') cannot be created in address space '__attribute__((address_space(1)))'}}
}

// A by-value call result is a prvalue and does not keep the return type's
// address space, so the temporary materialized for a member call is a plain S.
as1_S make();
const_as1_S makeConst();
as1_S &getRef();

void calls() {
  make().f();
  makeConst().cf(); // const is retained.

  // A reference names an object in the address space.
  getRef().f(); // expected-error {{cannot initialize object parameter of type 'S' with an expression of type 'as1_S' (aka '__attribute__((address_space(1))) S')}}
}

// A reference return binds to an object in the address space.
as1_S &retRef() { return gs; }
