// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -std=c++17 -x c++ -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -std=c++17 -x objective-c++ -ast-dump %s | FileCheck %s

// Exercises the generic sugar fallback in
// `ConstructDynamicBoundType::VisitType` (clang/lib/Sema/SemaDeclAttr.cpp): a
// `Type` sugar class that has no dedicated `Visit*Type` method is handled by
// peeling one layer of sugar and re-dispatching `Visit`.
//
// C++-only trigger: an alias-template specialization is a
// `TemplateSpecializationType`, which has no dedicated visitor (companion C
// file uses `BTFTagAttributedType`, which is ignored in C++). Here `NN<int>`
// desugars to `int * _Nonnull`, i.e. an `AttributedType` wrapping a pointer.
//
// The constructed `CountAttributedType` shows two things:
//   * The `TemplateSpecializationType` sugar (`NN<int>`) is *stripped*. The
//     fallback returns `Visit(Desugared)` and never rebuilds the peeled sugar
//     (see the FIXME at `VisitTypedefType`, rdar://185140320). Were the
//     fallback absent, dispatch would land in the reject branch and the
//     attribute application would fail instead of producing a valid type, so
//     the surviving `__counted_by` here is what justifies the fallback.
//   * The `AttributedType` underneath (`_Nonnull`) is *preserved*, because
//     `VisitAttributedType` rebuilds it. That is the contrast: only the sugar
//     handled by the fallback is dropped.

#include <ptrcheck.h>

template <class T> using NN = T * _Nonnull;

struct S {
  int n;

  // Without a bounds attribute the alias-template sugar is present as written.
  // CHECK: plain 'NN<int>':'int *'
  NN<int> plain;

  // FIXME: The `NN<int>` sugar shouldn't be dropped (rdar://185244036).
  // With `__counted_by` the `NN<int>` sugar is gone but `_Nonnull` survives.
  // CHECK: withct 'int * __counted_by(n) _Nonnull':'int *'
  NN<int> withct __counted_by(n);
};
