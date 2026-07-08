// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}
// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnretainedCallArgsChecker -verify %s

#include "objc-mock-types.h"

void consume_cf(CFMutableArrayRef);
void consume_obj(SomeObj *);

namespace call_args_const_retainptr_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const RetainPtr<SomeObj> m_constObj;
  RetainPtr<SomeObj> m_obj;
};

void Foo::bar() {
  [m_constObj doWork]; // no-warning
  [m_obj doWork]; // expected-warning{{Receiver is unretained and unsafe}}
}

} // namespace call_args_const_retainptr_member

namespace call_args_const_retainptr_cf_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const RetainPtr<CFMutableArrayRef> m_cf1;
  RetainPtr<CFMutableArrayRef> m_cf2;
};

void Foo::bar() {
  consume_cf(m_cf1.get()); // no-warning
  consume_cf(m_cf2.get()); // expected-warning{{Call argument is unretained and unsafe}}
}

} // namespace call_args_const_retainptr_cf_member

namespace call_args_const_retainptr_struct_member {

struct Bar {
  Bar();
  void baz();

  const RetainPtr<SomeObj> m_constObj;
  RetainPtr<SomeObj> m_obj;
};

void Bar::baz() {
  [m_constObj doWork]; // no-warning
  [m_obj doWork]; // expected-warning{{Receiver is unretained and unsafe}}
}

} // namespace call_args_const_retainptr_struct_member

namespace call_args_const_retainptr_cf_struct_member {

struct Bar {
  Bar();
  void baz();

  const RetainPtr<CFMutableArrayRef> m_cf1;
  RetainPtr<CFMutableArrayRef> m_cf2;
};

void Bar::baz() {
  consume_cf(m_cf1.get()); // no-warning
  consume_cf(m_cf2.get()); // expected-warning{{Call argument is unretained and unsafe}}
}

} // namespace call_args_const_retainptr_cf_struct_member

namespace call_args_const_retainptr_get_as_objc_arg {

class Foo {
public:
  Foo();
  void bar();

private:
  const RetainPtr<SomeObj> m_constObj;
  RetainPtr<SomeObj> m_obj;
};

void Foo::bar() {
  consume_obj(m_constObj.get()); // no-warning
  consume_obj(m_obj.get()); // expected-warning{{Call argument is unretained and unsafe}}
}

} // namespace call_args_const_retainptr_get_as_objc_arg

namespace call_args_const_retainptr_implicit_conv_arg {

class Foo {
public:
  Foo();
  void bar();

private:
  const RetainPtr<SomeObj> m_constObj;
  RetainPtr<SomeObj> m_obj;
};

void Foo::bar() {
  consume_obj(m_constObj); // no-warning
  consume_obj(m_obj); // expected-warning{{Call argument is unretained and unsafe}}
}

} // namespace call_args_const_retainptr_implicit_conv_arg

namespace call_args_const_osobjectptr_member {

class Foo {
public:
  Foo();
  void bar();

private:
  const OSObjectPtr<SomeObj *> m_constObj;
  OSObjectPtr<SomeObj *> m_obj;
};

void Foo::bar() {
  consume_obj(m_constObj.get()); // no-warning
  consume_obj(m_obj.get()); // expected-warning{{Call argument is unretained and unsafe}}
}

} // namespace call_args_const_osobjectptr_member

namespace call_args_const_osobjectptr_receiver {

class Foo {
public:
  Foo();
  void bar();

private:
  const OSObjectPtr<SomeObj *> m_constObj;
  OSObjectPtr<SomeObj *> m_obj;
};

void Foo::bar() {
  [m_constObj doWork]; // no-warning
  [m_obj doWork]; // expected-warning{{Receiver is unretained and unsafe}}
}

} // namespace call_args_const_osobjectptr_receiver

namespace call_args_retainptr_local {

void testLocal(SomeObj *input) {
  RetainPtr<SomeObj> localObj = input;
  [localObj doWork]; // no-warning
  consume_obj(localObj.get()); // no-warning
  consume_cf(RetainPtr<CFMutableArrayRef>().get()); // no-warning
}

} // namespace call_args_retainptr_local

namespace call_args_retainptr_protected_member {

class Foo {
public:
  Foo();
  void bar();

private:
  RetainPtr<SomeObj> m_obj;
};

void Foo::bar() {
  auto protectedObj = m_obj;
  [protectedObj doWork]; // no-warning (local copy is safe)
}

} // namespace call_args_retainptr_protected_member
