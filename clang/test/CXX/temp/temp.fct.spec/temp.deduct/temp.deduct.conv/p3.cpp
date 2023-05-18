// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
struct AnyPtr {
  template<typename T>
  operator T*() const;
};

// If A is a cv-qualified type, the top level cv-qualifiers of A's type
// are ignored for type deduction.
void test_cvquals(AnyPtr ap) {
  int* const ip = ap;
  const float * const volatile fp = ap;
}

// If A is a reference type, the type referred to by A is used for
// type deduction.
void test_ref_arg(AnyPtr ap) {
  const int* const &ip = ap;
  double * const &dp = ap;
}

struct AnyRef {
  template<typename T>
  operator T&() const;
};

void test_ref_param(AnyRef ar) {
  int &ir = ar;
  const float &fr = ar;
  int i = ar;
}
