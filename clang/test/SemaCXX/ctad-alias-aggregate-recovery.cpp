// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

// A prior fatal diagnostic makes InstantiatingTemplate invalid, so alias
// aggregate-guide synthesis returns null. The old code then llvm::cast a
// TypeAliasDecl to CXXRecordDecl. That SIGSEGVs on clang 22.1.8. Rel+Asserts
// trunk may not crash: the cast is UB and can be optimized into a null return.

template <class T> struct S {
  T v;
};
template <class T>
using U = S<T>;
#include "this_header_does_not_exist.h" // expected-error {{'this_header_does_not_exist.h' file not found}}
U x{0};
