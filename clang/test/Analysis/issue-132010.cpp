// RUN: %clang_analyze_cc1 -analyzer-config aggressive-binary-operation-simplification=true -analyzer-checker=alpha.cplusplus.MismatchedIterator -analyzer-output text -verify %s

// expected-no-diagnostics

#include "Inputs/system-header-simulator-cxx.h"

void f()
{
    std::list<int> l;
    std::unordered_set<int> us;
    us.insert(l.cbegin(), l.cend()); // no warning
}
