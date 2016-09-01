// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

extern int array[1];

template <typename>
class C {
  enum { D };
public:
  template <typename A> void foo1() {
    extern int array[((int)C<A>::k > (int)D) ? 1 : -1];
  }
};

template<>
class C<int> {
public:
  const static int k = 2;
};

void foo2() {
  C<char> c;
  c.foo1<int>();
}

template<int n>
void foo3() {
  extern int array[n ? 1 : -1];
}

void foo4() {
  foo3<5>();
}
