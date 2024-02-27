// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

class RefCounted {
public:
  void ref() const;
  void deref() const;
};

class Object {
public:
  void ref() const;
  void deref() const;
  void someFunction(RefCounted&);
};

RefPtr<Object> object();
RefPtr<RefCounted> protectedTargetObject();

void testFunction() {
  if (RefPtr obj = object())
    obj->someFunction(*protectedTargetObject());
}
