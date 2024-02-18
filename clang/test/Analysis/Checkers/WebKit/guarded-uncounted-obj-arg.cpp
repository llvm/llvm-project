// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

class Number {
public:
  Number();
  ~Number();
};

int someFunction();

class RefCounted {
public:
  void ref() const;
  void deref() const;
  void someMethod();
  bool isDerived() const { return false; }
};

class Derived : public RefCounted {
public:
  void otherMethod();
  bool isDerived() const { return true; }
};

template <typename S>
struct TypeCastTraits<const Derived, S> {
  static bool isOfType(const S &arg) { return arg.isDerived(); }
};

RefCounted *object();
void someFunction(const RefCounted&);

void test() {
  RefPtr obj = object();

  if (!obj->isDerived()) {
    auto* base = obj.get();
    base->someMethod();
  }

  if (auto *derived = dynamicDowncast<Derived>(*obj))
    derived->otherMethod();

  while (auto *derived = dynamicDowncast<Derived>(*obj)) {
    derived->otherMethod();
    break;
  }

  for (auto *derived = dynamicDowncast<Derived>(*obj); true; ) {
    derived->otherMethod();
    break;
  }

  do {
    auto *derived = dynamicDowncast<Derived>(*obj);
    derived->otherMethod();
  } while (0);

  auto* derived = dynamicDowncast<Derived>(*obj);
  derived->otherMethod();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

