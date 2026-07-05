// RUN: %clang_cc1 -fsyntax-only -verify %s
// Test instantiation of member functions of class templates defined out-of-line
template<typename T, typename U>
struct X0 {
  void f(T *t, const U &u);
  void f(T *);
};

template<typename T, typename U>
void X0<T, U>::f(T *t, const U &u) {
  *t = u; // expected-error{{indirection not permitted on operand of type 'void *'}} expected-error{{not assignable}}
}

void test_f(X0<float, int> xfi, X0<void, int> xvi, float *fp, void *vp, int i) {
  xfi.f(fp, i);
  xvi.f(vp, i); // expected-note{{instantiation}}
}
