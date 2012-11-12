// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: the "note" should be down at the call site!
template<typename T> void f1(T*) throw(T); // expected-error{{incomplete type 'Incomplete' is not allowed in exception specification}} \
                         // expected-note{{instantiation of}}
struct Incomplete; // expected-note{{forward}}

void test_f1(Incomplete *incomplete_p, int *int_p) {
  f1(int_p);
  f1(incomplete_p); 
}
