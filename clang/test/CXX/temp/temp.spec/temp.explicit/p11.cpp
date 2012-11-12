// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

class X {
  template <typename T> class Y {};
};

class A {
  class B {};
  class C {};
};

// C++0x [temp.explicit] 14.7.2/11:
//   The usual access checking rules do not apply to names used to specify
//   explicit instantiations.
template class X::Y<A::B>;

// As an extension, this rule is applied to explicit specializations as well.
template <> class X::Y<A::C> {};
