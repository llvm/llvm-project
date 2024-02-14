// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

class RefCounted {
public:
  void ref() const;
  void deref() const;
  void someFunction();
};

RefCounted* refCountedObj();

void test()
{
  refCountedObj()->someFunction();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}
