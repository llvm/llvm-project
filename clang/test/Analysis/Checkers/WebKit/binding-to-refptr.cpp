// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s -std=c++2c

#include "mock-types.h"

class Node : public CanMakeWeakPtr<Node> {
public:
    Node* nextSibling() const;

    void ref() const;
    void deref() const;
};

template <class A, class B> struct pair {
  A a;
  B b;
  template <unsigned I> requires ( I == 0 ) A& get();
  template <unsigned I> requires ( I == 1 ) B& get();
};

namespace std {
    template <class> struct tuple_size;
    template <unsigned, class> struct tuple_element;
    template <class A, class B> struct tuple_size<::pair<A, B>> { static constexpr int value = 2; };
    template <class A, class B> struct tuple_element<0, ::pair<A, B>> { using type = A; };
    template <class A, class B> struct tuple_element<1, ::pair<A, B>> { using type = B; };
}

pair<RefPtr<Node>, RefPtr<Node>> &getPair();

static void testUnpackedAssignment() {
    auto [a, b] = getPair();
    a->nextSibling();
}

pair<RefPtr<Node>, WeakPtr<Node>> getStrongWeakPair();
pair<WeakPtr<Node>, RefPtr<Node>> getWeakStrongPair();
static void testUnpackedAssignmentWithWeak() {
  auto [a, b] = getStrongWeakPair();
  a->nextSibling();
  b->nextSibling();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe [alpha.webkit.UncountedCallArgsChecker]}}
  auto [c, d] = getWeakStrongPair();
  c->nextSibling();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe [alpha.webkit.UncountedCallArgsChecker]}}
  d->nextSibling();
}
