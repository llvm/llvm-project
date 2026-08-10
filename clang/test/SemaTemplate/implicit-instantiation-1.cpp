// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T, typename U>
struct X {
  T f(T x, U y) { return x + y; }

  unsigned g(T x, U y) { return sizeof(f(x, y)); }
};

void test(X<int, int> *xii, X<int*, int> *xpi, X<int, int*> *xip) {
  (void)xii->f(1, 2);
  (void)xpi->f(0, 2);
  (void)sizeof(xip->f(2, 0)); // okay: does not instantiate
  (void)xip->g(2, 0); // okay: does not instantiate
}

template<typename T, typename U>
T add(T t, U u) {
  return t + u; // expected-error{{invalid operands}}
}

void test_add(char *cp, int i, int *ip) {
  char* cp2 = add(cp, i);
  add(cp, cp); // expected-note{{instantiation of}}
  (void)sizeof(add(ip, ip));
}
