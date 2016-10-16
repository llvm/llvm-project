// RUN: %clang_cc1 -fsyntax-only %s -verify

struct AnyT {
  template<typename T>
  operator T();
};

void test_cvqual_ref(AnyT any) {
  const int &cir = any;  
}

struct AnyThreeLevelPtr {
  template<typename T>
  operator T***() const {
    T x = 0; // expected-note 2{{declared const here}}
    x = 0; // expected-error 2{{const-qualified type}}
    T ***p;
    return p;
  }
};

struct X { };

void test_deduce_with_qual(AnyThreeLevelPtr a3) {
  int * const * const * const ip1 = a3;
  // FIXME: This is wrong; we are supposed to deduce 'T = int' here.
  const int * const * const * const ip2 = a3; // expected-note {{instantiation of}}
  // This one is correct, though.
  const double * * * ip3 = a3; // expected-note {{instantiation of}}
}

struct AnyPtrMem {
  template<typename Class, typename T>
  operator T Class::*() const
  {
    // This is correct: we don't need a qualification conversion here, so we
    // deduce 'T = const float'.
    T x = 0; // expected-note {{declared const here}}
    x = 0; // expected-error {{const-qualified type}}
    return 0;
  }
};

void test_deduce_ptrmem_with_qual(AnyPtrMem apm) {
  const float X::* pm = apm; // expected-note {{instantiation of}}
}

struct TwoLevelPtrMem {
  template<typename Class1, typename Class2, typename T>
  operator T Class1::*Class2::*() const
  {
    T x = 0; // expected-note 2{{declared const here}}
    x = 0; // expected-error 2{{const-qualified type}}
    return 0;
  }
};

void test_deduce_two_level_ptrmem_with_qual(TwoLevelPtrMem apm) {
  // FIXME: This is wrong: we should deduce T = 'float'
  const float X::* const X::* pm2 = apm; // expected-note {{instantiation of}}
  // This is correct: we don't need a qualification conversion, so we directly
  // deduce T = 'const double'
  const double X::* X::* pm1 = apm; // expected-note {{instantiation of}}
}
