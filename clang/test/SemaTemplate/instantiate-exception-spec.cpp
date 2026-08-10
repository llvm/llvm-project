// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -verify %std_cxx98-14 %s -DERRORS
// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -emit-llvm-only %std_cxx98-14 %s

#ifdef ERRORS
template<typename T> void f1(T*) throw(T); // expected-error{{incomplete type 'Incomplete' is not allowed in exception specification}}
struct Incomplete; // expected-note{{forward}}

void test_f1(Incomplete *incomplete_p, int *int_p) {
  f1(int_p);
  f1(incomplete_p); // expected-note{{instantiation of}}
}
#endif

template<typename T> void f(void (*p)() throw(T)) {
#ifdef ERRORS
  void (*q)() throw(char) = p; // expected-error {{target exception spec}}

  extern void (*p2)() throw(T);
  void (*q2)() throw(char) = p2; // expected-error {{target exception spec}}

  extern void (*p3)() throw(char);
  void (*q3)() throw(T) = p3; // expected-error {{target exception spec}}

  void (*q4)() throw(T) = p2; // ok
#endif
  p();
}
void g() { f<int>(0); } // expected-note {{instantiation of}}
