// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedLocalVarsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void someFunction();

namespace local_vars_const_checkedptr_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const CheckedPtr<CheckedObj> m_obj1;
  CheckedPtr<CheckedObj> m_obj2;
};

void Foo::bar() {
  auto* obj1 = m_obj1.get();
  obj1->method();
  auto* obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  obj2->method();
}

} // namespace local_vars_const_checkedptr_member

namespace local_vars_const_checkedref_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const CheckedRef<CheckedObj> m_obj1;
  CheckedRef<CheckedObj> m_obj2;
};

void Foo::bar() {
  auto& obj1 = m_obj1.get();
  obj1.method();
  auto& obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  obj2.method();
}

} // namespace local_vars_const_ref_member

namespace call_args_const_unique_ptr {

class Foo {
public:
  Foo();
  void bar();

private:
  const std::unique_ptr<CheckedObj> m_obj1;
  std::unique_ptr<CheckedObj> m_obj2;
};

void Foo::bar() {
  auto* obj1 = m_obj1.get();
  obj1->method();
  auto* obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  obj2->method();
}

} // namespace call_args_const_unique_ptr
