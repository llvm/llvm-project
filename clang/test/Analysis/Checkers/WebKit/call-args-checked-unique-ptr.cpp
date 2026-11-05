// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

#include "mock-types.h"

void consume(CheckedObj*);
void consumeRef(CheckedObj&);

namespace local_unique_ptr {

void foo() {
  std::unique_ptr<CheckedObj> obj;
  obj->method();
  (*obj).method();
  consumeRef(*obj);
  consume(&*obj);
  consume(obj.get());
  obj.get()->method();
}

void bar(std::unique_ptr<CheckedObj>& obj, const std::unique_ptr<CheckedObj>& constObj) {
  obj->method();
  consume(obj.get());
  constObj->method();
  consume(constObj.get());
}

} // namespace local_unique_ptr

namespace local_unique_ref {

void foo(CheckedObj& target) {
  UniqueRef<CheckedObj> obj(target);
  obj->method();
  obj.get().method();
  consumeRef(obj.get());
  consumeRef(obj);
  consume(&obj.get());
}

} // namespace local_unique_ref

namespace member_unique_ptr {

class Foo {
public:
  void bar();

private:
  const std::unique_ptr<CheckedObj> m_constObj;
  std::unique_ptr<CheckedObj> m_obj;
};

void Foo::bar() {
  consume(m_constObj.get());
  m_constObj.get()->method();
  consume(m_obj.get());
  // expected-warning@-1{{Function argument 'this->m_obj.get()' (to 'consume') is a raw pointer to CheckedPtr-capable type 'CheckedObj'}}
  m_obj.get()->method();
  // expected-warning@-1{{Function argument 'this->m_obj.get()' (parameter 'this' to 'CheckedObj::method') is a raw pointer to CheckedPtr-capable type 'CheckedObj'}}
}

} // namespace member_unique_ptr
