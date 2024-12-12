// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

namespace std {
}

namespace call_args_const_refptr_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const RefPtr<RefCountable> m_obj1;
  RefPtr<RefCountable> m_obj2;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

} // namespace call_args_const_refptr_member

namespace call_args_const_ref_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const Ref<RefCountable> m_obj1;
  Ref<RefCountable> m_obj2;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

} // namespace call_args_const_ref_member

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
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

} // namespace call_args_const_unique_ptr

namespace call_args_const_unique_ref {

class Foo {
public:
  Foo();
  void bar();

private:
  const UniqueRef<RefCountable> m_obj1;
  UniqueRef<RefCountable> m_obj2;
};

void Foo::bar() {
  m_obj1->method();
  m_obj2->method();
  // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}

} // namespace call_args_const_unique_ref
