// RUN: %clang_cc1 -fsyntax-only -verify %s

// POC diagnostics for __builtin_to_chars / __builtin_from_chars.

void to_chars_checks(char *p, const char *cp, int v, int base) {
  (void)__builtin_to_chars(p, p, v, 10);    // ok
  (void)__builtin_to_chars(p, p, v, base);  // ok, runtime base

  __builtin_to_chars(p, p, v); // expected-error {{too few arguments}}
  __builtin_to_chars(p, p, v, 10, p); // expected-error {{too many arguments}}

  // Buffer must be a pointer to non-const char.
  (void)__builtin_to_chars(v, p, v, 10);  // expected-error {{must be a pointer to non-const 'char'}}
  (void)__builtin_to_chars(cp, p, v, 10); // expected-error {{must be a pointer to non-const 'char'}}

  // Value must be an integer.
  (void)__builtin_to_chars(p, p, p, 10); // expected-error {{value argument to '__builtin_to_chars' must be an integer type}}

  // Constant base must be in [2, 36].
  (void)__builtin_to_chars(p, p, v, 1);  // expected-error {{base argument to '__builtin_to_chars' must be between 2 and 36}}
  (void)__builtin_to_chars(p, p, v, 37); // expected-error {{base argument to '__builtin_to_chars' must be between 2 and 36}}
}

void from_chars_checks(const char *cp, char *p, int *out, const int *cout, int base) {
  int ec;
  (void)__builtin_from_chars(cp, cp, out, 10, &ec);   // ok
  (void)__builtin_from_chars(p, p, out, base, &ec);   // ok, non-const buffer is fine

  __builtin_from_chars(cp, cp, out, 10); // expected-error {{too few arguments}}

  // Value must be a pointer to a non-const integer.
  (void)__builtin_from_chars(cp, cp, cout, 10, &ec); // expected-error {{value argument to '__builtin_from_chars' must be a pointer to a non-const integer type}}
  (void)__builtin_from_chars(cp, cp, base, 10, &ec); // expected-error {{value argument to '__builtin_from_chars' must be a pointer to a non-const integer type}}

  // ec must be a pointer to a non-const integer.
  (void)__builtin_from_chars(cp, cp, out, 10, base); // expected-error {{must be a pointer to a non-const integer type}}

  // Constant base range.
  (void)__builtin_from_chars(cp, cp, out, 0, &ec); // expected-error {{base argument to '__builtin_from_chars' must be between 2 and 36}}
}
