// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

struct A {
  unsigned x : 1;
};

// FIXME: static_cast<T&&> of a bit-field should bind to an independent
// temporary; currently Sema binds directly to the bit-field so r aliases a.x.
constexpr unsigned f1() {
  A a{0};
  unsigned &&r = static_cast<unsigned &&>(a.x);
  r = 7;
  a.x = 0;
  return r;
}
static_assert(f1() == 7); // expected-error {{static assertion failed due to requirement 'f1() == 7'}} \
                          // expected-note {{expression evaluates to '0 == 7'}}

// FIXME: the temporary's lifetime should expire at the end of the
// full-expression; reading through p after that point must be rejected.
// Currently no diagnostic is produced because the reference aliases a.x.
// This test uses -std=c++20: since C++23 (P2266), 'return a;' in q() is an
// xvalue and cannot bind to the 'auto &' return type, regardless of bit-fields.
constexpr auto &q(auto &&a) { return a; }
constexpr unsigned f2() {
  A a{0};
  unsigned *p;
  *(p = &q(static_cast<unsigned &&>(a.x))) = 1; // FIXME-expected-note {{temporary created here}}
  return *p; // FIXME-expected-note {{read of object outside its lifetime}}
}
static_assert(f2() == 1); // FIXME-expected-error {{static assertion expression is not an integral constant expression}} \
                          // FIXME-expected-note {{in call to 'f2()'}}

// FIXME: cv-qualification from static_cast<const T&&> must be preserved on the
// materialized temporary; const_cast cannot enable modification through it.
// f3() must never produce a constant expression. Currently no error is emitted
// because the direct binding lets the modification silently succeed.
// When fixed, this function should be diagnosed as never producing a constant
// expression, with a note pointing at the modification of the const temporary.
// The new interpreter has a separate bug and may not emit the note.
constexpr unsigned f3() { // FIXME-expected-error {{never produces a constant expression}}
  A a{0};
  unsigned &&r = const_cast<unsigned &&>(static_cast<const unsigned &&>(a.x));
  r = 0; // FIXME-expected-note {{modification of object of const-qualified type 'const unsigned int'}}
  return a.x;
}
// FIXME: this should be rejected once the const-qualified temporary is
// materialized. Currently it is accepted because the modification goes through
// the bit-field itself.
static_assert(f3() == 0);
