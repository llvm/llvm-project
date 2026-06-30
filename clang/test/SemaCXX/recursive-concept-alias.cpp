// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template<typename T> concept C1 = C1<T>; // expected-error {{a concept definition cannot refer to itself}} expected-note {{here}}