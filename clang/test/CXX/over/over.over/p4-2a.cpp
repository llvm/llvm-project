// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T, typename U>
constexpr static bool is_same_v = false;

template<typename T>
constexpr static bool is_same_v<T, T> = true;

template<typename T>
concept AtLeast2 = sizeof(T) >= 2;

template<typename T>
concept AtMost8 = sizeof(T) <= 8;

template<typename T>
struct S {
static int foo() requires AtLeast2<long> && AtMost8<long> {
  return 0;
}

static double foo() requires AtLeast2<char> {
  return 0.0;
}

static char bar() requires AtLeast2<char> {
  return 1.0;
}

static short bar() requires AtLeast2<long> && AtMost8<long> {
  return 0.0;
}

static int bar() requires AtMost8<long> && AtLeast2<long> {
  return 0.0;
}

static char baz() requires AtLeast2<char> {
  return 1.0;
}

static short baz() requires AtLeast2<long> && AtMost8<long> {
  return 0.0;
}

static int baz() requires AtMost8<long> && AtLeast2<long> {
  return 0.0;
}

static long baz() requires AtMost8<long> && AtLeast2<long> && AtLeast2<short> {
  return 3.0;
}
};

void a() {
  static_assert(is_same_v<decltype(&S<int>::foo), int(*)()>);
  static_assert(is_same_v<decltype(&S<int>::bar), long(*)()>);
  // expected-error@-1{{reference to overloaded function could not be resolved; did you mean to call it?}}
  static_assert(is_same_v<decltype(&S<int>::baz), long(*)()>);
}
