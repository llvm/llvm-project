// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

#include "mock-types.h"

namespace call_args_const_checkedptr_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const CheckedPtr<CheckedObj> m_obj1;
  CheckedPtr<CheckedObj> m_obj2;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
}

} // namespace call_args_const_checkedptr_member

namespace call_args_const_checkedref_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const CheckedRef<CheckedObj> m_obj1;
  CheckedRef<CheckedObj> m_obj2;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
}

} // namespace call_args_const_checkedref_member

namespace call_args_const_unique_ptr {

class Foo {
public:
  Foo();
  void bar();

  CheckedObj& ensureObj3() {
    if (!m_obj3)
      const_cast<std::unique_ptr<CheckedObj>&>(m_obj3) = new CheckedObj;
    return *m_obj3;
  }

  CheckedObj& badEnsureObj4() {
    if (!m_obj4)
      const_cast<std::unique_ptr<CheckedObj>&>(m_obj4) = new CheckedObj;
    if (auto* next = m_obj4->next())
      return *next;
    return *m_obj4;
  }

  CheckedObj* ensureObj5() {
    if (!m_obj5)
      const_cast<std::unique_ptr<CheckedObj>&>(m_obj5) = new CheckedObj;
    if (m_obj5->next())
      return nullptr;
    return m_obj5.get();
  }

  CheckedObj* ensureObj6() {
    if (!m_obj6)
      const_cast<std::unique_ptr<CheckedObj>&>(m_obj6) = new CheckedObj;
    if (m_obj6->next())
      return (CheckedObj *)0;
    return m_obj6.get();
  }

private:
  const std::unique_ptr<CheckedObj> m_obj1;
  std::unique_ptr<CheckedObj> m_obj2;
  const std::unique_ptr<CheckedObj> m_obj3;
  const std::unique_ptr<CheckedObj> m_obj4;
  const std::unique_ptr<CheckedObj> m_obj5;
  const std::unique_ptr<CheckedObj> m_obj6;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
  ensureObj3().method();
  badEnsureObj4().method();
  // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
  ensureObj5()->method();
  ensureObj6()->method();
}

} // namespace call_args_const_unique_ptr
