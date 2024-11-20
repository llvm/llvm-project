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

private:
  const std::unique_ptr<CheckedObj> m_obj1;
  std::unique_ptr<CheckedObj> m_obj2;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
}

} // namespace call_args_const_unique_ptr
