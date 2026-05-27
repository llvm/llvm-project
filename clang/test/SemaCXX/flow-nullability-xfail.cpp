// Known flow-sensitive nullability gaps.
//
// These are intentional expected failures so we can keep the missing behavior
// visible without breaking the main consolidated test files.
//
// XFAIL: *
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 -fcxx-exceptions %s -verify

struct Node {
  int value;
  Node * _Nullable next;
};

#pragma clang assume_nonnull begin

// Crubit tracks instantiated operator[] return nullability through the
// templated alias/reference path. nullable-clang currently does not.
template <typename T>
struct TemplateVec {
  using reference = T &;
  reference operator[](int);
};

void xfail_template_operator_subscript() {
  TemplateVec<Node *> nonnull;
  TemplateVec<Node * _Nullable> nullable;

  nonnull[0]->value = 1; // OK
  nullable[0]->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Crubit also preserves substituted nullability through temporary
// materialization and templated identity wrappers. nullable-clang currently
// loses that nullability.
template <typename T>
T identity(const T &);

template <typename T>
struct Holder {
  T get();
};

void xfail_template_identity_materialization(Holder<Node *> &nonnull_holder,
                                             Holder<Node * _Nullable> &nullable_holder) {
  identity<Holder<Node *>>(nonnull_holder).get()->value = 1; // OK
  identity<Holder<Node * _Nullable>>(nullable_holder).get()->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

#pragma clang assume_nonnull end
