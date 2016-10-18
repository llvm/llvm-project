// RUN: %clang_cc1 -std=c++1z -verify %s

template<typename T> void redecl1() noexcept(noexcept(T())) {} // expected-note {{previous}}
template<typename T> void redecl1() noexcept(noexcept(T())); // ok, same type
template<typename T> void redecl1() noexcept(noexcept(T())) {} // expected-error {{redefinition}}

template<bool A, bool B> void redecl2() noexcept(A); // expected-note {{previous}}
template<bool A, bool B> void redecl2() noexcept(B); // expected-error {{conflicting types}}

// These have the same canonical type.
// FIXME: It's not clear whether this is supposed to be valid.
template<typename A, typename B> void redecl3() throw(A);
template<typename A, typename B> void redecl3() throw(B);
