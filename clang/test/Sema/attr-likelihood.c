// RUN: %clang_cc1 %s -fsyntax-only -verify

void g(void) {
  if (1)
    [[clang::likely]] {}
}
void m(void) {
  [[clang::likely]] int x = 42; // expected-error {{'likely' attribute cannot be applied to a declaration}}

  if (x)
    [[clang::unlikely]] {}
  if (x) {
    [[clang::unlikely]];
  }
  switch (x) {
  case 1:
    [[clang::likely]] {}
    break;
    [[clang::likely]] case 2 : case 3 : {}
    break;
  }

  do {
    [[clang::unlikely]];
  } while (x);
  do
    [[clang::unlikely]] {}
  while (x);
  do { // expected-note {{to match this 'do'}}
  }
  [[clang::unlikely]] while (x); // expected-error {{expected 'while' in do/while loop}}
  for (;;)
    [[clang::unlikely]] {}
  for (;;) {
    [[clang::unlikely]];
  }
  while (x)
    [[clang::unlikely]] {}
  while (x) {
    [[clang::unlikely]];
  }

  if (x)
    goto lbl;

  // FIXME: allow the attribute on the label
  [[clang::unlikely]] lbl : // expected-error {{'unlikely' attribute cannot be applied to a declaration}}
  [[clang::likely]] x = x + 1;

  [[clang::likely]]++ x;
}
