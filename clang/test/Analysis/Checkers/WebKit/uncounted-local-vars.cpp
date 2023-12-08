// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"

namespace raw_ptr {
void foo() {
  RefCountable *bar;
  // FIXME: later on we might warn on uninitialized vars too
}

void bar(RefCountable *) {}
} // namespace raw_ptr

namespace reference {
void foo_ref() {
  RefCountable automatic;
  RefCountable &bar = automatic;
  // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
}

void bar_ref(RefCountable &) {}
} // namespace reference

namespace guardian_scopes {
void foo1() {
  RefPtr<RefCountable> foo;
  { RefCountable *bar = foo.get(); }
}

void foo2() {
  RefPtr<RefCountable> foo;
  // missing embedded scope here
  RefCountable *bar = foo.get();
  // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
}

void foo3() {
  RefPtr<RefCountable> foo;
  {
    { RefCountable *bar = foo.get(); }
  }
}

void foo4() {
  {
    RefPtr<RefCountable> foo;
    { RefCountable *bar = foo.get(); }
  }
}
} // namespace guardian_scopes

namespace auto_keyword {
class Foo {
  RefCountable *provide_ref_ctnbl() { return nullptr; }

  void evil_func() {
    RefCountable *bar = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    auto *baz = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'baz' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    auto *baz2 = this->provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'baz2' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  }
};
} // namespace auto_keyword

namespace guardian_casts {
void foo1() {
  RefPtr<RefCountable> foo;
  { RefCountable *bar = downcast<RefCountable>(foo.get()); }
}

void foo2() {
  RefPtr<RefCountable> foo;
  {
    RefCountable *bar =
        static_cast<RefCountable *>(downcast<RefCountable>(foo.get()));
  }
}
} // namespace guardian_casts

namespace guardian_ref_conversion_operator {
void foo() {
  Ref<RefCountable> rc;
  { RefCountable &rr = rc; }
}
} // namespace guardian_ref_conversion_operator

namespace ignore_for_if {
RefCountable *provide_ref_ctnbl() { return nullptr; }

void foo() {
  // no warnings
  if (RefCountable *a = provide_ref_ctnbl()) { }
  for (RefCountable *a = provide_ref_ctnbl(); a != nullptr;) { }
  RefCountable *array[1];
  for (RefCountable *a : array) { }
}
} // namespace ignore_for_if
