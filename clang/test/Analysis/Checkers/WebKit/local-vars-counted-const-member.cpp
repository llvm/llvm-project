// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void someFunction();

namespace local_vars_const_refptr_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const RefPtr<RefCountable> m_obj1;
  RefPtr<RefCountable> m_obj2;
};

void Foo::bar() {
  auto* obj1 = m_obj1.get();
  obj1->method();
  auto* obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  obj2->method();
}

} // namespace local_vars_const_refptr_member

namespace local_vars_const_ref_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const Ref<RefCountable> m_obj1;
  Ref<RefCountable> m_obj2;
};

void Foo::bar() {
  auto& obj1 = m_obj1.get();
  obj1.method();
  auto& obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  obj2.method();
}

} // namespace local_vars_const_ref_member

namespace call_args_const_unique_ptr {

class Foo {
public:
  Foo();
  void bar();

private:
  const std::unique_ptr<RefCountable> m_obj1;
  std::unique_ptr<RefCountable> m_obj2;
};

void Foo::bar() {
  auto* obj1 = m_obj1.get();
  obj1->method();
  auto* obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  obj2->method();
}

} // namespace call_args_const_unique_ptr
