// RUN: %clang_cc1 -fsyntax-only -pedantic-errors -verify %s

template<typename T> struct A {
  template<typename U> struct B {
    // FIXME: The standard does not seem to consider non-friend elaborated-type-specifiers that
    // declare partial specializations/explicit specializations/explicit instantiations to be
    // declarative, see https://lists.isocpp.org/core/2024/01/15325.php
    struct C;
    template<typename V> struct D;

    void f();
    template<typename V> void g();

    static int x;
    template<typename V> static int y;

    enum class E;
  };
};

template<typename T>
template<typename U>
struct A<T>::template B<U>::C { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
struct A<int>::template B<bool>::C; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
struct A<int>::template B<bool>::C { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::template B<U>::D<V*>; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::B<U>::template D<V**>; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::template B<U>::D { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::template B<U>::D<V*> { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
struct A<T>::B<U>::template D<V**> { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::template B<bool>::D; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
struct A<int>::template B<bool>::D<short>; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
struct A<int>::B<bool>::template D<long>; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::template B<bool>::D<V*>; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::B<bool>::template D<V**>; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::template B<bool>::D { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
struct A<int>::template B<bool>::D<short> { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
struct A<int>::B<bool>::template D<long> { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::template B<bool>::D<V*> { }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
struct A<int>::B<bool>::template D<V**> { }; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
void A<T>::template B<U>::f() { } // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
void A<int>::template B<bool>::f() { } // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
void A<T>::template B<U>::g() { } // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
void A<int>::B<bool>::template g<short>() { } // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
void A<int>::template B<bool>::g<long>() { } // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
void A<int>::template B<bool>::g() { } // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
int A<T>::template B<U>::x = 0; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
int A<T>::template B<U>::y = 0; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
int A<T>::template B<U>::y<V*> = 0; // expected-error{{'template' cannot be used after a declarative}}

template<typename T>
template<typename U>
template<typename V>
int A<T>::B<U>::template y<V**> = 0; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
int A<int>::template B<bool>::y = 0; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
int A<int>::template B<bool>::y<short> = 0; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<>
int A<int>::B<bool>::template y<long> = 0; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
int A<int>::template B<bool>::y<V*> = 0; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
template<typename V>
int A<int>::B<bool>::template y<V**> = 0; // expected-error{{'template' cannot be used after a declarative}}
template<typename T>
template<typename U>
enum class A<T>::template B<U>::E { a }; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
enum class A<int>::template B<bool>::E; // expected-error{{'template' cannot be used after a declarative}}

template<>
template<>
enum class A<int>::template B<bool>::E { a }; // expected-error{{'template' cannot be used after a declarative}}

// FIXME: We don't call Sema::diagnoseQualifiedDeclaration for friend declarations right now
template<typename T>
struct F {
  // FIXME: f should be assumed to name a template per [temp.names] p3.4
  friend void T::f<int>();
  // expected-error@-1{{use 'template' keyword to treat 'f' as a dependent template name}}
  // expected-error@-2{{no candidate function template was found for}}

  // FIXME: We should diagnose the presence of 'template' here
  friend void T::template f<int>(); // expected-error{{no candidate function template was found for}}
  friend void T::template U<int>::f();

  // These should be allowed
  friend class T::template U<int>;
  friend class T::template U<int>::V;
};
