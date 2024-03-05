// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

class Node {
public:
    Node* nextSibling() const;

    void ref() const;
    void deref() const;
};

static void removeDetachedChildren(Node* firstChild)
{
    for (RefPtr child = firstChild; child; child = child->nextSibling());
}
