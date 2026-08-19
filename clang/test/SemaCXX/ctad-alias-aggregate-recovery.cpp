// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

// Aggregate CTAD for an alias template after error recovery must not crash.
// DeclareAggregateDeductionGuideFromInitList used to fall through and
// llvm::cast the alias's TypeAliasDecl to CXXRecordDecl.

template <class T> struct S {
  T v;
};
template <class T>
using U = S<T>;
#include "this_header_does_not_exist.h" // expected-error {{'this_header_does_not_exist.h' file not found}}
U x{0};
