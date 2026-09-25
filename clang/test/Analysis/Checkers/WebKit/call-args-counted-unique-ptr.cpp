// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

void consume(RefCountable*);
void consumeRef(RefCountable&);

namespace local_unique_ptr {

void foo() {
  std::unique_ptr<RefCountable> obj = RefCountable::makeUnique();
  obj->method();
  (*obj).method();
  consumeRef(*obj);
  consume(&*obj);
  consume(obj.get());
  obj.get()->method();
}

void bar(std::unique_ptr<RefCountable>& obj, const std::unique_ptr<RefCountable>& constObj) {
  obj->method();
  consume(obj.get());
  constObj->method();
  consume(constObj.get());
}

} // namespace local_unique_ptr

namespace local_unique_ref {

void foo(RefCountable& target) {
  UniqueRef<RefCountable> obj(target);
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
  const std::unique_ptr<RefCountable> m_constObj;
  std::unique_ptr<RefCountable> m_obj;
};

void Foo::bar() {
  consume(m_constObj.get());
  m_constObj.get()->method();
  consume(m_obj.get());
  // expected-warning@-1{{Function argument 'this->m_obj.get()' (to 'consume') is a raw pointer to RefPtr-capable type 'RefCountable'}}
  m_obj.get()->method();
  // expected-warning@-1{{Function argument 'this->m_obj.get()' (parameter 'this' to 'RefCountable::method') is a raw pointer to RefPtr-capable type 'RefCountable'}}
}

} // namespace member_unique_ptr
