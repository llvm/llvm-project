// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

struct Obj {
  void ref() const;
  void deref() const;

  void someFunction();
};

template<typename T> class Wrapper {
public:
  T obj();
};

static void foo(Wrapper<Ref<Obj>>&& wrapper)
{
  wrapper.obj()->someFunction();
}
