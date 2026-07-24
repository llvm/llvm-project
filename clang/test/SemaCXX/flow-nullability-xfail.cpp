// Known flow-sensitive nullability gaps in template-substituted nullability.
//
// Converted from an unconditional expected-failure to assertions of CURRENT (incomplete) behavior, so
// the file PASSES today and any regression in the parts that DO work becomes
// visible. Each gap is documented with a FIXME describing what the analysis
// SHOULD do once the missing modeling lands. See
// flow-nullability-checker-gaps.cpp for the same pattern.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 -fcxx-exceptions -Wno-unused-value %s -verify

// Every case below is a MISSED warning (a gap): the analysis currently emits
// nothing, which is exactly what makes this file pass today. Each gap is marked
// with a FIXME. If any gap gets fixed (a warning starts firing) this directive
// fails and flags the case to convert to a positive assertion.
// expected-no-diagnostics

struct Node {
  int value;
  Node * _Nullable next;
};

#pragma clang assume_nonnull begin

// Crubit tracks instantiated operator[] return nullability through the
// templated alias/reference path. nullable-clang currently does not, so the
// nullable subscript dereference below is NOT flagged.
template <typename T>
struct TemplateVec {
  using reference = T &;
  reference operator[](int);
};

void xfail_template_operator_subscript() {
  TemplateVec<Node *> nonnull;
  TemplateVec<Node * _Nullable> nullable;

  nonnull[0]->value = 1; // OK
  // FIXME: should warn{{dereference of nullable pointer}} once nullability is
  // tracked through the templated `reference` alias of operator[]. Currently
  // the substituted _Nullable is lost, so no warning fires.
  nullable[0]->value = 1; // no warning (gap)
}

// Crubit also preserves substituted nullability through temporary
// materialization and templated identity wrappers. nullable-clang currently
// loses that nullability, so neither dereference below is flagged.
template <typename T>
T identity(const T &);

template <typename T>
struct Holder {
  T get();
};

void xfail_template_identity_materialization(Holder<Node *> &nonnull_holder,
                                             Holder<Node * _Nullable> &nullable_holder) {
  identity<Holder<Node *>>(nonnull_holder).get()->value = 1; // OK
  // FIXME: should warn{{dereference of nullable pointer}} once substituted
  // nullability survives temporary materialization through identity<>().get().
  // Currently it is lost, so no warning fires.
  identity<Holder<Node * _Nullable>>(nullable_holder).get()->value = 1; // no warning (gap)
}

#pragma clang assume_nonnull end
