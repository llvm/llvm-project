// RUN: %clang_cc1 -fsyntax-only -Wunknown-attribute-namespace=foo,bar -std=c23 -verify %s

[[foo::a(b((c)) d(e((f)))), foo::g(h k)]]
int f1(void) {
  return 0;
}

[[bar::a(b((c)) d(e((f)))), bar::g(h k)]]
int f2(void) {
  return 0;
}

[[baz::a(b((c)) d(e((f)))), baz::g(h k)]] // expected-warning {{unknown attribute 'a' ignored}} \
                                          // expected-warning {{unknown attribute 'g' ignored}}
int f3(void) {
  return 0;
}


[[clang::a(b((c)) d(e((f)))), clang::g(h k)]] // expected-warning {{unknown attribute 'a' ignored}} \
                                              // expected-warning {{unknown attribute 'g' ignored}}
int f4(void) {
  return 0;
}

[[clang::noinline]]
int f5(void) {
  return 0;
}
