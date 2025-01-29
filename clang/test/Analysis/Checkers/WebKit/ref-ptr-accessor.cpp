// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

void someFunction(RefCountable*);

void testFunction()
{
    Ref item = RefCountable::create();
    someFunction(item.ptr());
}
