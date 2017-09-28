// RUN: %clang_analyze_cc1 -w -x c++ -analyzer-checker=core -analyzer-output=text -analyzer-eagerly-assume -verify %s

namespace pr34731 {
int b;
class c {
  class B {
   public:
    double ***d;
    B();
  };
  void e(double **, int);
  void f(B &, int &);
};

// Properly track the null pointer in the array field back to the default
// constructor of 'h'.
void c::f(B &g, int &i) {
  e(g.d[9], i); // expected-warning{{Array access (via field 'd') results in a null pointer dereference}}
                // expected-note@-1{{Array access (via field 'd') results in a null pointer dereference}}
  B h, a; // expected-note{{Value assigned to 'h.d'}}
  a.d == __null; // expected-note{{Assuming the condition is true}}
  a.d != h.d; // expected-note{{Assuming pointer value is null}}
  f(h, b); // expected-note{{Calling 'c::f'}}
}
}
