// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

decltype([]()->decltype(this) { }) a; // expected-error {{invalid use of 'this' outside of a non-static member function}}

