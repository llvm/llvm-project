// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

#include "mock-types.h"

RefCountableAndCheckable* makeObj();
CheckedRef<RefCountableAndCheckable> makeObjChecked();
void someFunction(RefCountableAndCheckable*);

namespace call_args_unchecked_uncounted {

static void foo() {
  someFunction(makeObj());
  // expected-warning@-1{{Function argument 'makeObj()' (to 'someFunction') is a raw pointer to CheckedPtr-capable type 'RefCountableAndCheckable'}}
}

} // namespace call_args_unchecked_uncounted

namespace call_args_checked {

static void foo() {
  CheckedPtr<RefCountableAndCheckable> ptr = makeObj();
  someFunction(ptr.get());
}

static void bar() {
  someFunction(CheckedPtr { makeObj() }.get());
}

static void baz() {
  someFunction(makeObjChecked().ptr());
}

} // namespace call_args_checked

namespace call_args_member {

void consume(CheckedObj&);

struct WrapperObj {
  CheckedObj checked;
  CheckedObj& checkedRef;
  void foo() {
    consume(checked);
    consume(checkedRef);
    // expected-warning@-1{{Function argument 'this->checkedRef' (to 'call_args_member::consume') is a raw reference to CheckedPtr-capable type 'CheckedObj'}}
  }
  void bar(WrapperObj& other) {
    consume(other.checked);
    // expected-warning@-1{{Function argument 'other.checked' (to 'call_args_member::consume') is a raw reference to CheckedPtr-capable type 'CheckedObj'}}
  }
};

} // namespace call_args_checked

namespace call_args_default {

void someFunction(RefCountableAndCheckable* = makeObj());
// expected-warning@-1{{Function argument 'makeObj()' (to 'call_args_default::someFunction') is a raw pointer to CheckedPtr-capable type 'RefCountableAndCheckable'}}
void otherFunction(RefCountableAndCheckable* = makeObjChecked().ptr());

void foo() {
  someFunction();
  otherFunction();
}

}

namespace call_args_checked_assignment {

CheckedObj* provide();
void foo() {
  CheckedPtr<CheckedObj> ptr;
  ptr = provide();
}

}

namespace call_with_std_move {

void consume(CheckedObj&&);
void foo(CheckedObj&& obj) {
  consume(std::move(obj));
  consume(WTF::move(obj));
}

}
