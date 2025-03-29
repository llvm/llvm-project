// RUN: %clang_cc1 -verify -std=c++98 %s
// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: %clang_cc1 -verify -std=c++14 %s
// RUN: %clang_cc1 -verify -std=c++17 %s
// RUN: %clang_cc1 -verify -std=c++20 %s
// RUN: %clang_cc1 -verify -std=c++23 %s
// RUN: %clang_cc1 -verify -std=c++2c %s

// https://github.com/llvm/llvm-project/issues/67914

template < typename, int >
struct Mask;

template < int, class >
struct conditional {
  using type = Mask< int, 16 >; // expected-warning 0+ {{}}
};

template < class _Then >
struct conditional< 0, _Then > {
  using type = _Then; // expected-warning 0+ {{}}
};

template < int _Bp, class, class _Then >
using conditional_t = typename conditional< _Bp, _Then >::type; // expected-warning 0+ {{}}

template < typename, int >
struct Array;

template < typename, int, bool, typename >
struct StaticArrayImpl;

template < typename Value_, int Size_ >
struct Mask : StaticArrayImpl< Value_, Size_, 1, Mask< Value_, Size_ > > { // expected-note 0+ {{}}
  template < typename T1 >
  Mask(T1) {} // expected-note 0+ {{}}
};

template < typename T >
void load(typename T::MaskType mask) {
  T::load_(mask); // expected-note 0+ {{}}
}

template < typename Value_, int IsMask_, typename Derived_ >
struct StaticArrayImpl< Value_, 32, IsMask_, Derived_ > {
  using Array1 = conditional_t< IsMask_, void, Array< Value_, 16 > >; // expected-warning 0+ {{}}
  
  template < typename Mask >
  static Derived_ load_(Mask mask) {
    return Derived_{load< Array1 >(mask.a1), Mask{}}; // expected-error 0+ {{}}
  }

  Array1 a1;
};

template < typename Derived_ >
struct KMaskBase;

template < typename Derived_ >
struct StaticArrayImpl< float, 16, 0, Derived_ > {
  template < typename Mask >
  static Derived_ load_(Mask mask);
};

template < typename Derived_ >
struct StaticArrayImpl< float, 16, 1, Mask< float, 16 > > : KMaskBase< Derived_ > {}; // expected-error 0+ {{}}

template < typename Derived_ >
struct StaticArrayImpl< int, 16, 1, Derived_ > {};

template < typename Value_, int Size_ >
struct Array : StaticArrayImpl< Value_, Size_, 0, Array< Value_, Size_ > > {
  using MaskType = Mask< Value_, Size_ >; // expected-warning 0+ {{}}
};

void test11_load_masked() {
  load< Array< float, 32 > >{} == 0; // expected-error 0+ {{}} expected-warning 0+ {{}} expected-note 0+ {{}}
}
