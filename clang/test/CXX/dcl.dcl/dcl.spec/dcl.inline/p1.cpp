// RUN: %clang_cc1 -std=c++1z -verify %s

inline int f(); // ok
inline int n; // ok

inline typedef int t; // expected-error {{'inline' can only appear on functions and non-local variables}}
inline struct S {}; // expected-error {{'inline' can only appear on functions and non-local variables}}
inline struct T {} s; // ok
