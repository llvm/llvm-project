// RUN: %clang_cc1 -std=c++20 -Wno-register -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

// A global register variable (GNU extension) has no memory address, so C++
// should reject taking its address, as C does.
register unsigned long sp asm("rsp");

unsigned long *take_address() {
  return &sp; // expected-error {{address of register variable requested}}
}

// Reading the value is fine, only taking the address is rejected.
unsigned long read_value() {
  return sp;
}

void bind_reference() {
  unsigned long &ref = sp; // expected-error {{non-const reference cannot bind to global register variable}}
  (void)ref;
}

// Taking the address of an ordinary register variable stays valid in C++
// (c++03 7.1.1p3), so the check must not apply to it.
void local_register_ok() {
  register int x = 0;
  int *q = &x;
  (void)q;
}

// A non-const reference can't bind to a global register variable: there is
// no object to refer to.
unsigned long &bad_return() {
  return sp; // expected-error {{non-const reference cannot bind to global register variable}}
}

void takes_ref(unsigned long &x); // expected-note {{passing argument to parameter 'x' here}}
void bad_call() {
  takes_ref(sp); // expected-error {{non-const reference cannot bind to global register variable}}
}

// A const reference binds to a temporary copy of the register's value.
void const_ref_ok() {
  const unsigned long &r = sp;
  (void)r;
}

// Parentheses around the variable don't change the result.
void paren_ref() {
  unsigned long &r = (sp); // expected-error {{non-const reference cannot bind to global register variable}}
  (void)r;
}

void paren_const_ref_ok() {
  const unsigned long &r = (sp);
  (void)r;
}
