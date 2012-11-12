// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// C++0x [temp.local]p1:
//   Like normal (non-template) classes, class templates have an
//   injected-class-name (Clause 9). The injected-class-name can be used with
//   or without a template-argument-list. When it is used without
//   a template-argument-list, it is equivalent to the injected-class-name
//   followed by the template-parameters of the class template enclosed in <>.

template <typename T> struct X0 {
  X0();
  ~X0();
  X0 f(const X0&);
};

// Test non-type template parameters.
template <int N1, const int& N2, const int* N3> struct X1 {
  X1();
  ~X1();
  X1 f(const X1& x1a) { X1 x1b(x1a); return x1b; }
};

//   When it is used with a template-argument-list, it refers to the specified
//   class template specialization, which could be the current specialization
//   or another specialization.
// FIXME: Test this clause.

int i = 42;
void test() {
  X0<int> x0; (void)x0;
  X1<42, i, &i> x1; (void)x1;
}
