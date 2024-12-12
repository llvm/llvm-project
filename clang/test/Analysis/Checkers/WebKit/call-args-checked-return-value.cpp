// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

class Checkable {
public:
  void ref() const;
  void deref() const;
};

class Object {
public:
  void ref() const;
  void deref() const;
  void someFunction(Checkable&);
};

RefPtr<Object> object();
RefPtr<Checkable> protectedTargetObject();

void testFunction() {
  if (RefPtr obj = object())
    obj->someFunction(*protectedTargetObject());
}
