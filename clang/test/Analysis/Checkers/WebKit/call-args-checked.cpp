// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

RefCountableAndCheckable* makeObj();
CheckedRef<RefCountableAndCheckable> makeObjChecked();
void someFunction(RefCountableAndCheckable*);

namespace call_args_unchecked_uncounted {

static void foo() {
  someFunction(makeObj());
  // expected-warning@-1{{Call argument is uncounted and unsafe [alpha.webkit.UncountedCallArgsChecker]}}
}

} // namespace call_args_checked

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
// expected-warning@-1{{Call argument is uncounted and unsafe [alpha.webkit.UncountedCallArgsChecker]}}
void otherFunction(RefCountableAndCheckable* = makeObjChecked().ptr());

void foo() {
  someFunction();
  otherFunction();
}

}
