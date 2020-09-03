// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s
// UNSUPPORTED: ppc64be

/// Prevent use of all builtins.
void valid_attribute_all_1() __attribute__((no_builtin)) {}
void valid_attribute_all_2() __attribute__((no_builtin())) {}

/// Prevent use of specific builtins.
void valid_attribute_function() __attribute__((no_builtin("memcpy"))) {}
void valid_attribute_functions() __attribute__((no_builtin("memcpy"))) __attribute__((no_builtin("memcmp"))) {}

/// Many times the same builtin is fine.
void many_attribute_function_1() __attribute__((no_builtin)) __attribute__((no_builtin)) {}
void many_attribute_function_2() __attribute__((no_builtin("memcpy"))) __attribute__((no_builtin("memcpy"))) {}
void many_attribute_function_3() __attribute__((no_builtin("memcpy", "memcpy"))) {}
void many_attribute_function_4() __attribute__((no_builtin("memcpy", "memcpy"))) __attribute__((no_builtin("memcpy"))) {}

/// Invalid builtin name.
void invalid_builtin() __attribute__((no_builtin("not_a_builtin"))) {}
// expected-warning@-1 {{'not_a_builtin' is not a valid builtin name for 'no_builtin'}}

/// Can't use bare no_builtin with a named one.
void wildcard_and_functionname() __attribute__((no_builtin)) __attribute__((no_builtin("memcpy"))) {}
// expected-error@-1 {{empty 'no_builtin' cannot be composed with named ones}}

/// Can't attach attribute to a variable.
int __attribute__((no_builtin)) variable;
// expected-warning@-1 {{'no_builtin' attribute only applies to functions}}

/// Can't attach attribute to a declaration.
void nobuiltin_on_declaration() __attribute__((no_builtin));
// expected-error@-1 {{no_builtin attribute is permitted on definitions only}}

struct S {
  /// Can't attach attribute to a defaulted function,
  S()
  __attribute__((no_builtin)) = default;
  // expected-error@-1 {{no_builtin attribute has no effect on defaulted or deleted functions}}

  /// Can't attach attribute to a deleted function,
  S(const S &)
  __attribute__((no_builtin)) = delete;
  // expected-error@-1 {{no_builtin attribute has no effect on defaulted or deleted functions}}

  void whatever() __attribute__((no_builtin("memcpy")));
  // expected-error@-1 {{no_builtin attribute is permitted on definitions only}}
};

/// Can't attach attribute to an aliased function.
void alised_function() {}
void aliasing_function() __attribute__((no_builtin)) __attribute__((alias("alised_function")));
// expected-error@-1 {{no_builtin attribute is permitted on definitions only}}
