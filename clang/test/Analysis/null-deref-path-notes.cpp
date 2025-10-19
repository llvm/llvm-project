// RUN: %clang_analyze_cc1 -w -x c++ -analyzer-checker=core -analyzer-output=text -verify %s

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
  a.d != h.d; // expected-note{{Assuming 'a.d' is equal to 'h.d'}}
  f(h, b); // expected-note{{Calling 'c::f'}}
}
}

namespace GH124975 {
void no_crash_in_br_visitors(int *p) {
  if (p) {}
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{Taking false branch}}

  extern bool ExternLocalCoin;
  // expected-note@+2 {{Assuming 'ExternLocalCoin' is false}}
  // expected-note@+1 {{Taking false branch}}
  if (ExternLocalCoin)
    return;

  *p = 4;
  // expected-warning@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  // expected-note@-2    {{Dereference of null pointer (loaded from variable 'p')}}
}

// Thread local variables are implicitly static, so let's test them too.
void thread_local_alternative(int *p) {
  if (p) {}
  // expected-note@-1 {{Assuming 'p' is null}}
  // expected-note@-2 {{Taking false branch}}

  thread_local bool ThreadLocalCoin;
  // expected-note@+2 {{'ThreadLocalCoin' is false}}
  // expected-note@+1 {{Taking false branch}}
  if (ThreadLocalCoin)
    return;

  *p = 4;
  // expected-warning@-1 {{Dereference of null pointer (loaded from variable 'p')}}
  // expected-note@-2    {{Dereference of null pointer (loaded from variable 'p')}}
}
} // namespace GH124975
