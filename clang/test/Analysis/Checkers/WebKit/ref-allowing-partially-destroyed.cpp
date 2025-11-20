// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

template <typename T> struct RefAllowingPartiallyDestroyed {
  T *t;

  RefAllowingPartiallyDestroyed() : t{} {};
  RefAllowingPartiallyDestroyed(T &) {}
  T *get() { return t; }
  T *ptr() { return t; }
  T *operator->() { return t; }
  operator const T &() const { return *t; }
  operator T &() { return *t; }
};

template <typename T> struct RefPtrAllowingPartiallyDestroyed {
  T *t;

  RefPtrAllowingPartiallyDestroyed() : t(new T) {}
  RefPtrAllowingPartiallyDestroyed(T *t) : t(t) {}
  T *get() { return t; }
  T *operator->() { return t; }
  const T *operator->() const { return t; }
  T &operator*() { return *t; }
  RefPtrAllowingPartiallyDestroyed &operator=(T *) { return *this; }
  operator bool() { return t; }
};

class RefCounted {
public:
  void ref() const;
  void deref() const;
  void someFunction();
};

RefAllowingPartiallyDestroyed<RefCounted> object1();
RefPtrAllowingPartiallyDestroyed<RefCounted> object2();

void testFunction() {
  object1()->someFunction();
  object2()->someFunction();
}
