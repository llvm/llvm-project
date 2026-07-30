// RUN: %clang_cc1 -fsyntax-only -verify -Wvla %s

void test1(int n) { // expected-note {{here}}
  int v[n]; // expected-warning {{variable length array}} expected-note {{parameter 'n'}}
}

void test2(int n, int v[n]) { // expected-warning {{variable length array}} expected-note {{parameter 'n'}} expected-note {{here}}
}

void test3(int n, int v[n]); // expected-warning {{variable length array}} expected-note {{parameter 'n'}} expected-note {{here}}

template<typename T>
void test4(int n) { // expected-note {{here}}
  int v[n]; // expected-warning {{variable length array}} expected-note {{parameter 'n'}}
}

template<typename T>
void test5(int n, int v[n]) { // expected-warning {{variable length array}} expected-note {{parameter 'n'}} expected-note {{here}}
}

template<typename T>
void test6(int n, int v[n]); // expected-warning {{variable length array}} expected-note {{parameter 'n'}} expected-note {{here}}

template<typename T>
void test7(int n, T v[n]) { // expected-warning {{variable length array}} expected-note {{parameter 'n'}} expected-note {{here}}
}

