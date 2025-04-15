// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedLocalVarsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void someFunction();

namespace local_vars_const_checkedptr_member {

class Foo {
public:
  Foo();
  void bar();

  CheckedObj& ensureObj3() {
    if (!m_obj3)
      const_cast<CheckedPtr<CheckedObj>&>(m_obj3) = new CheckedObj;
    return *m_obj3;
  }

  CheckedObj& ensureObj4() {
    if (!m_obj4)
      const_cast<CheckedPtr<CheckedObj>&>(m_obj4) = new CheckedObj;
    if (auto* next = m_obj4->next()) {
      // expected-warning@-1{{Local variable 'next' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
      return *next;
    }
    return *m_obj4;
  }

  CheckedObj* ensureObj5() {
    if (!m_obj5)
      const_cast<CheckedPtr<CheckedObj>&>(m_obj5) = new CheckedObj;
    if (m_obj5->next())
      return nullptr;
    return m_obj5.get();
  }

private:
  const CheckedPtr<CheckedObj> m_obj1;
  CheckedPtr<CheckedObj> m_obj2;
  const CheckedPtr<CheckedObj> m_obj3;
  const CheckedPtr<CheckedObj> m_obj4;
  const CheckedPtr<CheckedObj> m_obj5;
};

void Foo::bar() {
  auto* obj1 = m_obj1.get();
  obj1->method();
  auto* obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  obj2->method();
  auto& obj3 = ensureObj3();
  obj3.method();
  auto& obj4 = ensureObj4();
  // expected-warning@-1{{Local variable 'obj4' is unchecked and unsafe [alpha.webkit.UncheckedLocalVarsChecker]}}
  obj4.method();
  auto* obj5 = ensureObj5();
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
