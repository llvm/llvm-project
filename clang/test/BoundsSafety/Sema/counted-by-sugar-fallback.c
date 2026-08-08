// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x c -ast-dump %s | FileCheck --check-prefixes=CHECK,FULL %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -ast-dump %s | FileCheck --check-prefixes=CHECK,ATTR %s

// Exercises the generic sugar fallback in
// `ConstructDynamicBoundType::VisitType` (clang/lib/Sema/SemaDeclAttr.cpp): a
// `Type` sugar class that has no dedicated `Visit*Type` method is handled by
// peeling one layer of sugar and re-dispatching `Visit`.
//
// C trigger: `btf_type_tag` forms a `BTFTagAttributedType`, which has no
// dedicated visitor (companion C++ file uses an alias-template
// `TemplateSpecializationType`; `btf_type_tag` is ignored in C++). Here it is
// the outermost sugar, wrapping a `_Nonnull` `AttributedType` over a pointer.
//
// The constructed `CountAttributedType` shows two things:
//   * The `BTFTagAttributedType` sugar is *stripped*. The fallback returns
//     `Visit(Desugared)` and never rebuilds the peeled sugar (see the FIXME at
//     `VisitTypedefType`, rdar://185140320). Were the fallback absent, dispatch
//     would land in the reject branch and the attribute application would fail
//     instead of producing a valid type, so the surviving `__counted_by` here
//     is what justifies the fallback.
//   * The `AttributedType` underneath (`_Nonnull`) is *preserved*, because
//     `VisitAttributedType` rebuilds it. That is the contrast: only the sugar
//     handled by the fallback is dropped.

#include <ptrcheck.h>

struct S {
  int n;

  // Without a bounds attribute both sugars are present, `btf_type_tag`
  // outermost. Identical in both modes.
  // CHECK: plain 'int * _Nonnull __attribute__((btf_type_tag("bt")))':'int *'
  int * _Nonnull __attribute__((btf_type_tag("bt"))) plain;

  // FIXME: `__attribute__((btf_type_tag("bt")))` should not be dropped
  // (rdar://185244036)
  // With `__counted_by` the `BTFTagAttributedType` is stripped by the fallback;
  // the `_Nonnull` `AttributedType` is preserved by `VisitAttributedType`.
  // FULL: withct 'int *__single __counted_by(n) _Nonnull':'int *__single'
  // ATTR: withct 'int * __counted_by(n) _Nonnull':'int *'
  int * _Nonnull __attribute__((btf_type_tag("bt"))) __counted_by(n) withct;
};
