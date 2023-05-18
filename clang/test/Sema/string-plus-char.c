// RUN: %clang_cc1 -fsyntax-only -verify %s

struct AB{const char *a; const char*b;};

const char *foo(const struct AB *ab) {
  return ab->a + 'b'; // expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}
}

void f(const char *s) {
  char *str = 0;
  char *str2 = str + 'c'; // expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}

  const char *constStr = s + 'c'; // expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}

  str = 'c' + str;// expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}

  char strArr[] = "foo";
  str = strArr + 'c'; // expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}
  char *strArr2[] = {"ac","dc"};
  str = strArr2[0] + 'c'; // expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}


  struct AB ab;
  constStr = foo(&ab) + 'c'; // expected-warning {{adding 'char' to a string pointer does not append to the string}} expected-note {{use array indexing to silence this warning}}

  // no-warning
  char c = 'c';
  str = str + c;
  str = c + str;
}
