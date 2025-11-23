// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

struct Clazz {
  template <typename T>
  static void templated_memfn();
};

// This must come before the 'templated_memfn' is defined!
static void instantiate() {
  Clazz::templated_memfn<int>();
}

template <typename T>
void Clazz::templated_memfn() {
  // When we report a bug in a function, we traverse the lexical decl context
  // of it while looking for suppression attributes to record what source
  // ranges should the suppression apply to.
  // In the past, that traversal didn't follow template instantiations, only
  // primary templates.
  [[clang::suppress]] clang_analyzer_warnIfReached(); // no-warning

}

namespace [[clang::suppress]]
suppressed_namespace {
  int foo() {
    int *x = 0;
    return *x;
  }

  int foo_forward();
}

int suppressed_namespace::foo_forward() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

// Another instance of the same namespace.
namespace suppressed_namespace {
  int bar() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}

void lambda() {
  [[clang::suppress]] {
    auto lam = []() {
      int *x = 0;
      return *x;
    };
  }
}

class [[clang::suppress]] SuppressedClass {
  int foo() {
    int *x = 0;
    return *x;
  }

  int bar();
};

int SuppressedClass::bar() {
  int *x = 0;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

class SuppressedMethodClass {
  [[clang::suppress]] int foo() {
    int *x = 0;
    return *x;
  }

  [[clang::suppress]] int bar1();
  int bar2();
};

int SuppressedMethodClass::bar1() {
  int *x = 0;
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

[[clang::suppress]]
int SuppressedMethodClass::bar2() {
  int *x = 0;
  return *x; // no-warning
}
