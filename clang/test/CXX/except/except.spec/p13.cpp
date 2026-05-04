// RUN: %clang_cc1 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

struct A {
  static constexpr bool x = true;
};

namespace N0 {

template<typename T, typename U>
void f(T, U) noexcept(T::y); // #1

template<typename T, typename U> // #2
void f(T, U*) noexcept(T::x);

// Deduction should succeed for both candidates, and #2 should be selected as the primary template.
// Only the exception specification of #2 should be instantiated.
template<>
void f(A, int*) noexcept;

}

namespace N1 {

template<typename T, typename U>
void f(T, U) noexcept(T::x); // #1

template<typename T, typename U>
void f(T, U*) noexcept(T::y); // #2
// expected-error@-1 {{no member named 'y' in 'A'}}

// Deduction should succeed for both candidates, and #2 should be selected as the primary template.
// Only the exception specification of #2 should be instantiated.
template<>
void f(A, int*) noexcept; // expected-error {{exception specification in declaration does not match previous declaration}}
                          // expected-note@-1 {{in instantiation of exception specification for 'f<A, int>' requested here}}
                          // expected-note@-2 {{previous declaration is here}}
}

namespace N2 {

template<typename T, typename U>
void f(T, U) noexcept(T::x);

template<typename T, typename U>
void f(T, U*) noexcept(T::x);

template<typename T, typename U>
void f(T, U**) noexcept(T::y); // expected-error {{no member named 'y' in 'A'}}

template<typename T, typename U>
void f(T, U***) noexcept(T::x);

template<>
void f(A, int*) noexcept; // expected-note {{previous declaration is here}}

template<>
void f(A, int*); // expected-error {{'f<A, int>' is missing exception specification 'noexcept'}}

template<>
void f(A, int**) noexcept; // expected-error {{exception specification in declaration does not match previous declaration}}
                           // expected-note@-1 {{in instantiation of exception specification for 'f<A, int>' requested here}}
                           // expected-note@-2 {{previous declaration is here}}

// FIXME: Exception specification is currently set to EST_None if instantiation fails.
template<>
void f(A, int**);

template<>
void f(A, int***) noexcept; // expected-note {{previous declaration is here}}

template<>
void f(A, int***); // expected-error {{'f<A, int>' is missing exception specification 'noexcept'}}

}

namespace N3 {

template<typename T, typename U>
void f(T, U) noexcept(T::y); // #1

template<typename T, typename U> // #2
void f(T, U*) noexcept(T::x);

// Deduction should succeed for both candidates, and #2 should be selected by overload resolution.
// Only the exception specification of #2 should be instantiated.
void (*x)(A, int*) = f;
}

namespace N4 {

template<typename T, typename U>
void f(T, U) noexcept(T::x); // #1

template<typename T, typename U>
void f(T, U*) noexcept(T::y); // #2
// expected-error@-1 {{no member named 'y' in 'A'}}

// Deduction should succeed for both candidates, and #2 should be selected by overload resolution.
// Only the exception specification of #2 should be instantiated.
void (*x)(A, int*) = f; // expected-note {{in instantiation of exception specification for 'f<A, int>' requested here}}
}
