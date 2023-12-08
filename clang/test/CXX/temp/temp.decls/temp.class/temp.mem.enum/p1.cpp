// RUN: %clang_cc1 -std=c++11 -verify %s

template<typename T> struct A {
  enum E : T; // expected-note {{here}}
  E v;
  E f() { return A::e1; } // expected-error {{no member named 'e1' in 'A<T>'}}
  E g() { return E::e1; }
  E h();
};

A<int> a;
A<int>::E a0 = A<int>().v;
int n = A<int>::E::e1; // expected-error {{implicit instantiation of undefined member}}

template<typename T> enum A<T>::E : T { e1, e2 }; // expected-note 2 {{declared here}}

// FIXME: Now that A<T>::E is defined, we are supposed to inject its enumerators
// into the already-instantiated class A<T>. This seems like a really bad idea,
// though, so we don't implement that, but what we do implement is inconsistent.
//
// Either do as the standard says, or only include enumerators lexically defined
// within the class in its scope.
A<int>::E a1 = A<int>::e1; // expected-error {{no member named 'e1' in 'A<int>'; did you mean simply 'e1'?}}

A<char>::E a2 = A<char>::e2;

template<typename T> typename A<T>::E A<T>::h() { return e2; }
A<short>::E a3 = A<short>().h();


template<typename T> struct B {
  enum class E;
  E v;
  E f() { return E::e1; }
  E g();
};

B<int> b;
B<int>::E b0 = B<int>().v;

template<typename T> enum class B<T>::E { e1, e2 };
B<int>::E b1 = B<int>::E::e1;

B<char>::E b2 = B<char>::E::e2;

template<typename T> typename B<T>::E B<T>::g() { return e2; }
B<short>::E b3 = B<short>().g();


// Enumeration members of class templates can be explicitly specialized. For
// unscoped enumerations, specializations must be defined before the primary
// template is, since otherwise the primary template will be implicitly
// instantiated when we parse the nested name specifier.
template<> enum A<long long>::E : long long { e3, e4 }; // expected-error {{explicit specialization of 'E' after instantiation}} expected-note {{first required here}}

template<> enum class B<long long>::E { e3, e4 };
B<long long>::E b4 = B<long long>::E::e4;

B<long>::E b5;
template<> enum class B<long>::E { e5 };
void fb5() { b5 = decltype(b5)::e5; }
B<long>::E b6 = B<long>::E::e5;


template<typename T> struct C {
  enum class E : T;
};

template<> enum class C<long long>::E : long long { e3, e4 };
C<long long>::E c0 = C<long long>::E::e3;

C<long>::E c1;
template<> enum class C<long>::E : long { e5 };
void fc1() { c1 = decltype(c1)::e5; }
C<long>::E c2 = C<long>::E::e5;

template<> enum class C<int>::E : int { e6 };
template<typename T> enum class C<T>::E : T { e0 };
C<int>::E c3 = C<int>::E::e6;
C<int>::E c4 = C<int>::E::e0; // expected-error {{no member named 'e0' in 'C<int>::E'}}


// Enumeration members can't be partially-specialized.
template<typename T> enum class B<T*>::E { e5, e6 }; // expected-error {{nested name specifier for a declaration cannot depend on a template parameter}}


// Explicit specializations can be forward-declared.
template<typename T>
struct D {
  enum class E { e1 };
};
template<> enum class D<int>::E;
D<int>::E d1 = D<int>::E::e1; // expected-error {{incomplete type 'D<int>::E'}}
template<> enum class D<int>::E { e2 };
D<int>::E d2 = D<int>::E::e2;
D<char>::E d3 = D<char>::E::e1; // expected-note {{first required here}}
D<char>::E d4 = D<char>::E::e2; // expected-error {{no member named 'e2' in 'D<char>::E'; did you mean simply 'e2'?}}
template<> enum class D<char>::E { e3 }; // expected-error {{explicit specialization of 'E' after instantiation}}

template<> enum class D<short>::E;
struct F {
  // Per C++11 [class.friend]p3, these friend declarations have no effect.
  // Only classes and functions can be friends.
  template<typename T> friend enum D<T>::E;
  template<> friend enum D<short>::E;

  template<> friend enum D<double>::E { e3 }; // expected-error {{cannot define a type in a friend declaration}}

private:
  static const int n = 1; // expected-note {{private here}}
};
template<> enum class D<short>::E {
  e = F::n // expected-error {{private member}}
};

class Access {
  friend class X;

  template<typename T>
  class Priv {
    friend class X;

    enum class E : T;
  };

  class S {
    typedef int N; // expected-note {{here}}
    static const int k = 3; // expected-note {{here}}

    friend class Priv<char>;
  };

  static const int k = 5;
};

template<> enum class Access::Priv<Access::S::N>::E
  : Access::S::N { // expected-error {{private member}}
  a = Access::k, // ok
  b = Access::S::k // expected-error {{private member}}
};

template<typename T> enum class Access::Priv<T>::E : T {
  c = Access::k,
  d = Access::S::k
};

class X {
  Access::Priv<int>::E a = Access::Priv<int>::E::a;
  Access::Priv<char>::E c = Access::Priv<char>::E::d;
  // FIXME: We should see an access error for this enumerator.
  Access::Priv<short>::E b = Access::Priv<short>::E::d;
};
