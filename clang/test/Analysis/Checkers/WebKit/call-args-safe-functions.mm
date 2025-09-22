// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "objc-mock-types.h"

CGImageRef provideImage();

Boolean cfe(CFTypeRef obj)
{
  return CFEqual(obj, provideImage());
}

