// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedLocalVarsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void someFunction();

namespace raw_ptr {
void foo() {
  CheckedObj *bar;
  // FIXME: later on we might warn on uninitialized vars too
}

void bar(CheckedObj *) {}
} // namespace raw_ptr

namespace reference {
void foo_ref() {
  CheckedObj automatic;
  CheckedObj &bar = automatic;
  // expected-warning@-1{{Local variable 'bar' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  someFunction();
  bar.method();
}

void foo_ref_trivial() {
  CheckedObj automatic;
  CheckedObj &bar = automatic;
}

void bar_ref(CheckedObj &) {}
} // namespace reference

namespace guardian_scopes {
void foo1() {
  CheckedPtr<CheckedObj> foo;
  { CheckedObj *bar = foo.get(); }
}

void foo2() {
  CheckedPtr<CheckedObj> foo;
  // missing embedded scope here
  CheckedObj *bar = foo.get();
  // expected-warning@-1{{Local variable 'bar' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  someFunction();
  bar->method();
}

void foo3() {
  CheckedPtr<CheckedObj> foo;
  {
    { CheckedObj *bar = foo.get(); }
  }
}

void foo4() {
  {
    CheckedPtr<CheckedObj> foo;
    { CheckedObj *bar = foo.get(); }
  }
}

void foo5() {
  CheckedPtr<CheckedObj> foo;
  auto* bar = foo.get();
  bar->trivial();
}

void foo6() {
  CheckedPtr<CheckedObj> foo;
  auto* bar = foo.get();
  // expected-warning@-1{{Local variable 'bar' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  bar->method();
}

struct SelfReferencingStruct {
  SelfReferencingStruct* ptr;
  CheckedObj* obj { nullptr };
};

void foo7(CheckedObj* obj) {
  SelfReferencingStruct bar = { &bar, obj };
  bar.obj->method();
}

} // namespace guardian_scopes

namespace auto_keyword {
class Foo {
  CheckedObj *provide_ref_ctnbl();

  void evil_func() {
    CheckedObj *bar = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'bar' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    auto *baz = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'baz' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    auto *baz2 = this->provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'baz2' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    [[clang::suppress]] auto *baz_suppressed = provide_ref_ctnbl(); // no-warning
  }

  void func() {
    CheckedObj *bar = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'bar' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    if (bar)
      bar->method();
  }
};
} // namespace auto_keyword

namespace guardian_casts {
void foo1() {
  CheckedPtr<CheckedObj> foo;
  {
    CheckedObj *bar = downcast<CheckedObj>(foo.get());
    bar->method();
  }
  foo->method();
}

void foo2() {
  CheckedPtr<CheckedObj> foo;
  {
    CheckedObj *bar =
        static_cast<CheckedObj *>(downcast<CheckedObj>(foo.get()));
    someFunction();
  }
}
} // namespace guardian_casts

namespace guardian_ref_conversion_operator {
void foo() {
  CheckedRef<CheckedObj> rc;
  {
    CheckedObj &rr = rc;
    rr.method();
    someFunction();
  }
}
} // namespace guardian_ref_conversion_operator

namespace ignore_for_if {
CheckedObj *provide_ref_ctnbl() { return nullptr; }

void foo() {
  // no warnings
  if (CheckedObj *a = provide_ref_ctnbl())
    a->trivial();
  for (CheckedObj *b = provide_ref_ctnbl(); b != nullptr;)
    b->trivial();
  CheckedObj *array[1];
  for (CheckedObj *c : array)
    c->trivial();
  while (CheckedObj *d = provide_ref_ctnbl())
    d->trivial();
  do {
    CheckedObj *e = provide_ref_ctnbl();
    e->trivial();
  } while (1);
  someFunction();
}

void bar() {
  if (CheckedObj *a = provide_ref_ctnbl()) {
    // expected-warning@-1{{Local variable 'a' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    a->method();    
  }
  for (CheckedObj *b = provide_ref_ctnbl(); b != nullptr;) {
    // expected-warning@-1{{Local variable 'b' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    b->method();
  }
  CheckedObj *array[1];
  for (CheckedObj *c : array) {
    // expected-warning@-1{{Local variable 'c' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    c->method();
  }

  while (CheckedObj *d = provide_ref_ctnbl()) {
    // expected-warning@-1{{Local variable 'd' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    d->method();
  }
  do {
    CheckedObj *e = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'e' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    e->method();
  } while (1);
  someFunction();
}

} // namespace ignore_for_if

namespace ignore_system_headers {

CheckedObj *provide_checkable();

void system_header() {
  localVar<CheckedObj>(provide_checkable);
}

} // ignore_system_headers

namespace conditional_op {
CheckedObj *provide_checkable();
bool bar();

void foo() {
  CheckedObj *a = bar() ? nullptr : provide_checkable();
  // expected-warning@-1{{Local variable 'a' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  CheckedPtr<CheckedObj> b = provide_checkable();
  {
    CheckedObj* c = bar() ? nullptr : b.get();
    c->method();
    CheckedObj* d = bar() ? b.get() : nullptr;
    d->method();
  }
}

} // namespace conditional_op

namespace local_assignment_basic {

CheckedObj *provide_checkable();

void foo(CheckedObj* a) {
  CheckedObj* b = a;
  // expected-warning@-1{{Local variable 'b' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  if (b->trivial())
    b = provide_checkable();
}

void bar(CheckedObj* a) {
  CheckedObj* b;
  // expected-warning@-1{{Local variable 'b' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  b = provide_checkable();
}

void baz() {
  CheckedPtr a = provide_checkable();
  {
    CheckedObj* b = a.get();
    // expected-warning@-1{{Local variable 'b' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    b = provide_checkable();
  }
}

} // namespace local_assignment_basic

namespace local_assignment_to_parameter {

CheckedObj *provide_checkable();
void someFunction();

void foo(CheckedObj* a) {
  a = provide_checkable();
  // expected-warning@-1{{Assignment to an unchecked parameter 'a' is unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  someFunction();
  a->method();
}

} // namespace local_assignment_to_parameter

namespace local_assignment_to_static_local {

CheckedObj *provide_checkable();
void someFunction();

void foo() {
  static CheckedObj* a = nullptr;
  // expected-warning@-1{{Static local variable 'a' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  a = provide_checkable();
  someFunction();
  a->method();
}

} // namespace local_assignment_to_static_local

namespace local_assignment_to_global {

CheckedObj *provide_ref_cntbl();
void someFunction();

CheckedObj* g_a = nullptr;
// expected-warning@-1{{Global variable 'local_assignment_to_global::g_a' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}

void foo() {
  g_a = provide_ref_cntbl();
  someFunction();
  g_a->method();
}

} // namespace local_assignment_to_global

namespace local_refcountable_checkable_object {

RefCountableAndCheckable* provide_obj();

void local_raw_ptr() {
  RefCountableAndCheckable* a = nullptr;
  // expected-warning@-1{{Local variable 'a' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  a = provide_obj();
  a->method();
}

void local_checked_ptr() {
  RefPtr<RefCountableAndCheckable> a = nullptr;
  a = provide_obj();
  a->method();
}

void local_var_with_guardian_checked_ptr() {
  RefPtr<RefCountableAndCheckable> a = provide_obj();
  {
    auto* b = a.get();
    b->method();
  }
}

void local_var_with_guardian_checked_ptr_with_assignment() {
  RefPtr<RefCountableAndCheckable> a = provide_obj();
  {
    RefCountableAndCheckable* b = a.get();
    // expected-warning@-1{{Local variable 'b' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
    b = provide_obj();
    b->method();
  }
}

void local_var_with_guardian_checked_ref() {
  Ref<RefCountableAndCheckable> a = *provide_obj();
  {
    RefCountableAndCheckable& b = a;
    b.method();
  }
}

void static_var() {
  static RefCountableAndCheckable* a = nullptr;
  // expected-warning@-1{{Static local variable 'a' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  a = provide_obj();
}

} // namespace local_refcountable_checkable_object
