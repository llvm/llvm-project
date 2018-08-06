// RUN: %clang_cc1 -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++11 -fsyntax-only %s -verify
// RUN: %clang_cc1 -std=c++17 -fsyntax-only %s -verify

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

namespace non_ptr_ref_cv_qual {
  template<typename Expected>
  struct ConvToT {
    template<typename T> operator T() {
      using Check = T;
      using Check = Expected;
    }
  };
  const int test_conv_to_t_1 = ConvToT<int>();
  // We intentionally deviate from [temp.deduct.conv]p4 here, and also remove
  // the top-level cv-quaifiers from A *after* removing the reference type, if
  // P is not also a reference type. This matches what other compilers are
  // doing, and is necessary to support real-world code.
  const int &test_conv_to_t_2 = ConvToT<int>();

  // Example code that would be broken by the standard's rule.
  struct Dest {};
  Dest d1a((ConvToT<Dest>()));
  Dest d1b = ConvToT<Dest>();
  Dest &d2 = (d1a = ConvToT<Dest>());

  template<typename Expected>
  struct ConvToTRef {
    template<typename T> operator T&() {
      using Check = T;
      using Check = Expected;
    }
  };
  const int test_conv_to_t_ref_1 = ConvToTRef<int>();
  const int &test_conv_to_t_ref_2 = ConvToTRef<const int>();

  Dest d3a((ConvToTRef<const Dest>())); // initialize the copy ctor parameter with 'const Dest&'
  Dest d3b = ConvToTRef<Dest>(); // convert to non-const T via [over.match.copy]/1.2
  Dest &d4 = (d3a = ConvToTRef<const Dest>());

  template<typename Expected>
  struct ConvToConstT {
    template<typename T> operator const T() {
      using Check = T;
      using Check = Expected;
    }
  };
  const int test_conv_to_const_t_1 = ConvToConstT<int>();
  const int &test_conv_to_const_t_2 = ConvToConstT<int>();

  template<typename Expected>
  struct ConvToConstTRef {
    template<typename T> operator const T&() {
      using Check = T;
      using Check = Expected;
    }
  };
  const int test_conv_to_const_t_ref_1 = ConvToConstTRef<int>();
  const int &test_conv_to_const_t_ref_2 = ConvToConstTRef<int>();

  template <typename T, int N> using Arr = T[N];
  struct ConvToArr {
    template <int N>
    operator Arr<int, N> &() {
      static_assert(N == 3, "");
    }
  };
  int (&test_conv_to_arr_1)[3] = ConvToArr(); // ok
  const int (&test_conv_to_arr_2)[3] = ConvToArr(); // ok, with qualification conversion

#if __cplusplus >= 201702L
  template<bool Noexcept, typename T, typename ...U> using Function = T(U...) noexcept(Noexcept);
  template<bool Noexcept> struct ConvToFunction {
    template <typename T, typename ...U> operator Function<Noexcept, T, U...>&(); // expected-note {{candidate}}
  };
  void (&fn1)(int) noexcept(false) = ConvToFunction<false>();
  void (&fn2)(int) noexcept(true)  = ConvToFunction<false>(); // expected-error {{no viable}}
  void (&fn3)(int) noexcept(false) = ConvToFunction<true>();
  void (&fn4)(int) noexcept(true)  = ConvToFunction<true>();

  struct ConvToFunctionDeducingNoexcept {
    template <bool Noexcept, typename T, typename ...U> operator Function<Noexcept, T, U...>&();
  };
  void (&fn5)(int) noexcept(false) = ConvToFunctionDeducingNoexcept();
  void (&fn6)(int) noexcept(true)  = ConvToFunctionDeducingNoexcept();
#endif
}
