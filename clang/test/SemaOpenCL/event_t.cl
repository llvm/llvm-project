// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

event_t glb_evt; // expected-error {{the 'event_t' type cannot be used to declare a program scope variable}}

constant struct evt_s {
  event_t evt; // expected-error {{the 'event_t' type cannot be used to declare a structure or union field}}
} evt_str = {0};

void foo(event_t evt); // expected-note {{passing argument to parameter 'evt' here}}

void kernel ker(event_t argevt) { // expected-error {{'event_t' cannot be used as the type of a kernel parameter}}
  event_t e;
  constant event_t const_evt; // expected-error {{the event_t type can only be used with __private address space qualifier}}
  foo(e);
  foo(0);
  foo(5); // expected-error {{passing 'int' to parameter of incompatible type 'event_t'}}
  foo((event_t)1); // expected-error {{cannot cast non-zero value '1' to 'event_t'}}
}

