// RUN: %clang_cc1 %s -fsyntax-only -verify
// RUN: %clang_cc1 %s -DPEDANTIC -pedantic -fsyntax-only -verify

#if PEDANTIC
void g() {
  if (true)
    [[likely]] {} // expected-warning {{use of the 'likely' attribute is a C++20 extension}}
  else
    [[unlikely]] {} // expected-warning {{use of the 'unlikely' attribute is a C++20 extension}}
}
#else
void a() {
  if (true)
    [[likely]]; // expected-warning {{conflicting attributes 'likely' are ignored}}
  else
    [[likely]]; // expected-note {{conflicting attribute is here}}
}

void b() {
  if (true)
    [[unlikely]]; // expected-warning {{conflicting attributes 'unlikely' are ignored}}
  else
    [[unlikely]]; // expected-note {{conflicting attribute is here}}
}

void c() {
  if (true)
    [[likely]];
}

void d() {
  if (true)
    [[unlikely]];
}

void g() {
  if (true)
    [[likely]] {}
  else
    [[unlikely]] {}
}

void h() {
  if (true)
    [[likely]] {}
  else {
  }
}

void i() {
  if (true)
    [[unlikely]] {}
  else {
  }
}

void j() {
  if (true) {
  } else
    [[likely]] {}
}

void k() {
  if (true) {
  } else
    [[likely]] {}
}

void l() {
  if (true)
    [[likely]] {}
  else
    [[unlikely]] if (false) [[likely]] {}
}

void m() {
  [[likely]] int x = 42; // expected-error {{'likely' attribute cannot be applied to a declaration}}

  if (x)
    [[unlikely]] {}
  if (x) {
    [[unlikely]];
  }
  switch (x) {
  case 1:
    [[likely]] {}
    break;
    [[likely]] case 2 : case 3 : {}
    break;
  }

  do {
    [[unlikely]];
  } while (x);
  do
    [[unlikely]] {}
  while (x);
  do { // expected-note {{to match this 'do'}}
  }
  [[unlikely]] while (x); // expected-error {{expected 'while' in do/while loop}}
  for (;;)
    [[unlikely]] {}
  for (;;) {
    [[unlikely]];
  }
  while (x)
    [[unlikely]] {}
  while (x) {
    [[unlikely]];
  }

  switch (x)
    [[unlikely]] {}

  if (x)
    goto lbl;

  // FIXME: allow the attribute on the label
  [[unlikely]] lbl : // expected-error {{'unlikely' attribute cannot be applied to a declaration}}
                     [[likely]] x = x + 1;

  [[likely]]++ x;
}

void n() [[likely]] // expected-error {{'likely' attribute cannot be applied to types}}
{
  try
    [[likely]] {} // expected-error {{expected '{'}}
  catch (...) [[likely]] { // expected-error {{expected expression}}
  }
}
#endif
