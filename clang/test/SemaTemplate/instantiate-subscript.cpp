// RUN: %clang_cc1 -fsyntax-only -verify %s


struct Sub0 {
  int &operator[](int);
};

struct Sub1 {
  long &operator[](long); // expected-note{{candidate function}}
};

struct ConvertibleToInt {
  operator int();
};

template<typename T, typename U, typename Result>
struct Subscript0 {
  void test(T t, U u) {
    Result &result = t[u]; // expected-error{{no viable overloaded operator[] for type}}
  }
};

template struct Subscript0<int*, int, int&>;
template struct Subscript0<Sub0, int, int&>;
template struct Subscript0<Sub1, ConvertibleToInt, long&>;
template struct Subscript0<Sub1, Sub0, long&>; // expected-note{{instantiation}}

// PR5345
template <typename T>
struct S {
  bool operator[](int n) const { return true; }
};

template <typename T>
void Foo(const S<int>& s, T x) {
  if (s[0]) {}
}

void Bar() {
  Foo(S<int>(), 0);
}
