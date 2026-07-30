// RUN: %clang_cc1 -verify -fsyntax-only -Wstring-conversion %s

void do_nothing(void);
void assert_error(void);

#define assert1(expr) \
  if (expr)           \
    do_nothing();     \
  else                \
  assert_error()

#define assert2(expr) \
  ((expr) ? do_nothing() : assert_error())

// Exception for common assert form.
void test1(void) {
  assert1(0 && "foo");
  assert1("foo" && 0);
  assert1(0 || "foo"); // expected-warning {{string literal}}
  assert1("foo"); // expected-warning {{string literal}}

  assert2(0 && "foo");
  assert2("foo" && 0);
  assert2(0 || "foo"); // expected-warning {{string literal}}
  assert2("foo"); // expected-warning {{string literal}}
}

void test2(void) {
  if ("hi") {}           // expected-warning {{string literal}}
  while ("hello") {}     // expected-warning {{string literal}}
  for (;"howdy";) {}     // expected-warning {{string literal}}
  do { } while ("hey");  // expected-warning {{string literal}}
  int x = "hey" ? 1 : 2; // expected-warning {{string literal}}
}
