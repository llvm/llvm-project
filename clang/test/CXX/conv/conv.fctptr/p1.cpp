// RUN: %clang_cc1 -std=c++1z -verify %s -triple x86_64-unknown-unknown

struct S;

typedef void Nothrow() noexcept;
typedef void Throw();

Nothrow *a;
Throw *b;
Nothrow S::*c;
Throw S::*d;

void test() {
  a = b; // expected-error {{incompatible function pointer types assigning to 'Nothrow *' (aka 'void (*)() noexcept') from 'Throw *' (aka 'void (*)()')}}
  b = a;
  c = d; // expected-error {{assigning to 'Nothrow S::*' from incompatible type 'Throw S::*': different exception specifications}}
  d = c;

  // Function pointer conversions do not combine properly with qualification conversions.
  // FIXME: This seems like a defect.
  Nothrow *const *pa = b; // expected-error {{cannot initialize}}
  Throw *const *pb = a; // expected-error {{cannot initialize}}
  Nothrow *const S::*pc = d; // expected-error {{cannot initialize}}
  Throw *const S::*pd = c; // expected-error {{cannot initialize}}
}

// ... The result is a pointer to the function.
void f() noexcept;
constexpr void (*p)() = &f;
static_assert(f == p);

struct S { void f() noexcept; };
constexpr void (S::*q)() = &S::f;
static_assert(q == &S::f);


namespace std_example {
  void (*p)();
  void (**pp)() noexcept = &p; // expected-error {{cannot initialize a variable of type 'void (**)() noexcept' with an rvalue of type 'void (**)()'}}

  struct S { typedef void (*p)(); operator p(); }; // expected-note {{candidate}}
  void (*q)() noexcept = S(); // expected-error {{no viable conversion from 'std_example::S' to 'void (*)() noexcept'}}
}
