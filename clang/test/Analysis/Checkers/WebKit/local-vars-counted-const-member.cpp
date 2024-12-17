// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void someFunction();

namespace local_vars_const_refptr_member {

class Foo {
public:
  Foo();
  void bar();

  RefCountable& ensureObj3() {
    if (!m_obj3)
      const_cast<RefPtr<RefCountable>&>(m_obj3) = RefCountable::create();
    return *m_obj3;
  }

  RefCountable& ensureObj4() {
    if (!m_obj4)
      const_cast<RefPtr<RefCountable>&>(m_obj4) = RefCountable::create();
    if (auto* next = m_obj4->next()) {
      // expected-warning@-1{{Local variable 'next' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
      return *next;
    }
    return *m_obj4;
  }

  RefCountable* ensureObj5() {
    if (!m_obj5)
      const_cast<RefPtr<RefCountable>&>(m_obj5) = RefCountable::create();
    if (m_obj5->next())
      return nullptr;
    return m_obj5.get();
  }

private:
  const RefPtr<RefCountable> m_obj1;
  RefPtr<RefCountable> m_obj2;
  const RefPtr<RefCountable> m_obj3;
  const RefPtr<RefCountable> m_obj4;
  const RefPtr<RefCountable> m_obj5;
};

void Foo::bar() {
  auto* obj1 = m_obj1.get();
  obj1->method();
  auto* obj2 = m_obj2.get();
  // expected-warning@-1{{Local variable 'obj2' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  obj2->method();
  auto& obj3 = ensureObj3();
  obj3.method();
  auto& obj4 = ensureObj4();
  // expected-warning@-1{{Local variable 'obj4' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  obj4.method();
  auto* obj5 = ensureObj5();
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
