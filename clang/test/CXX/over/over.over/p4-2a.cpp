// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T, typename U>
constexpr static bool is_same_v = false;

template<typename T>
constexpr static bool is_same_v<T, T> = true;

template<typename T>
concept AtLeast2 = sizeof(T) >= 2;

template<typename T>
concept AtMost8 = sizeof(T) <= 8;

int foo() requires AtLeast2<long> && AtMost8<long> {
  return 0;
}

double foo() requires AtLeast2<char> {
  return 0.0;
}

char bar() requires AtLeast2<char> { // expected-note {{possible target for call}}
  return 1.0;
}

short bar() requires AtLeast2<long> && AtMost8<long> {
// expected-note@-1{{possible target for call}}
// expected-note@-2{{candidate function}}
  return 0.0;
}

int bar() requires AtMost8<long> && AtLeast2<long> {
// expected-note@-1{{possible target for call}}
// expected-note@-2{{candidate function}}
  return 0.0;
}

char baz() requires AtLeast2<char> {
  return 1.0;
}

short baz() requires AtLeast2<long> && AtMost8<long> {
  return 0.0;
}

int baz() requires AtMost8<long> && AtLeast2<long> {
  return 0.0;
}

long baz() requires AtMost8<long> && AtLeast2<long> && AtLeast2<short> {
  return 3.0;
}

void a() {
  static_assert(is_same_v<decltype(&foo), int(*)()>);
  static_assert(is_same_v<decltype(&bar), long(*)()>);
  // expected-error@-1{{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
  // expected-error@-2{{call to 'bar' is ambiguous}}
  static_assert(is_same_v<decltype(&baz), long(*)()>);
}