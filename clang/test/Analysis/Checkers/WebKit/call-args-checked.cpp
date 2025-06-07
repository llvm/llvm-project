// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

#include "mock-types.h"

namespace std {

template <typename T> struct remove_reference {
  typedef T type;
};

template <typename T> struct remove_reference<T&> {
  typedef T type;
};

template<typename T> typename remove_reference<T>::type&& move(T&& t);

} // namespace std

RefCountableAndCheckable* makeObj();
CheckedRef<RefCountableAndCheckable> makeObjChecked();
void someFunction(RefCountableAndCheckable*);

namespace call_args_unchecked_uncounted {

static void foo() {
  someFunction(makeObj());
  // expected-warning@-1{{Call argument is unchecked and unsafe [alpha.webkit.UncheckedCallArgsChecker]}}
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

namespace call_args_default {

void someFunction(RefCountableAndCheckable* = makeObj());
// expected-warning@-1{{Call argument is unchecked and unsafe [alpha.webkit.UncheckedCallArgsChecker]}}
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
}

}
