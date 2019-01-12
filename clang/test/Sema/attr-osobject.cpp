// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S {
  __attribute__((os_returns_retained)) S* method_returns_retained() {
    return nullptr;
  }

  __attribute__((os_consumes_this)) void method_consumes_this();

  __attribute__((os_consumes_this)) static void rejected_on_static(); // expected-warning{{'os_consumes_this' attribute only applies to non-static member functions}}
};
__attribute__((os_returns_retained)) S *ret_retained() {
  return nullptr;
}

__attribute__((os_returns_retained)) S ret_retained_value() { // expected-warning{{'os_returns_retained' attribute only applies to functions that return a pointer}}
  return {};
}

__attribute__((os_returns_not_retained)) S *ret_not_retained() {
  return nullptr;
}

__attribute__((os_returns_not_retained)) S ret_not_retained_value() { // expected-warning{{'os_returns_not_retained' attribute only applies to functions that return a pointer}}
  return {};
}

void accept_consumed_arg(__attribute__((os_consumed)) S *arg) {}

void accept_consumed_arg_by_value(__attribute__((os_consumed)) S arg) {} // expected-warning{{os_consumed attribute only applies to pointer parameters}}

void accept_consumed_arg_no_extra_arg(__attribute__((os_consumed(10))) S *arg) {} // expected-error{{'os_consumed' attribute takes no arguments}}

struct __attribute__((os_consumed)) NoAttrOnStruct {}; // expected-warning{{'os_consumed' attribute only applies to parameters}}

__attribute__((os_returns_retained(10))) S* returns_retained_no_extra_arg() { // expected-error{{'os_returns_retained' attribute takes no arguments}}
  return nullptr;
}

struct __attribute__((os_returns_retained)) NoRetainAttrOnStruct {}; // expected-warning{{'os_returns_retained' attribute only applies to functions, Objective-C methods, Objective-C properties, and parameters}}

__attribute__((os_returns_not_retained(10))) S* os_returns_no_retained_no_extra_args( S *arg) { // expected-error{{'os_returns_not_retained' attribute takes no arguments}}
  return nullptr;
}

struct __attribute__((os_returns_not_retained)) NoNotRetainedAttrOnStruct {}; // expected-warning{{'os_returns_not_retained' attribute only applies to functions, Objective-C methods, Objective-C properties, and parameters}}

__attribute__((os_consumes_this)) void no_consumes_this_on_function() {} // expected-warning{{'os_consumes_this' attribute only applies to non-static member functions}}

void write_into_out_parameter(__attribute__((os_returns_retained)) S** out) {}

bool write_into_out_parameter_on_nonzero(__attribute__((os_returns_retained_on_non_zero)) S** out) {}

bool write_into_out_parameter_on_nonzero_invalid(__attribute__((os_returns_retained_on_non_zero)) S* out) {} // expected-warning{{'os_returns_retained_on_non_zero' attribute only applies to pointer/reference-to-OSObject-pointer parameters}}

bool write_into_out_parameter_on_zero(__attribute__((os_returns_retained_on_zero)) S** out) {}

bool write_into_out_parameter_on_zero_invalid(__attribute__((os_returns_retained_on_zero)) S* out) {} // expected-warning{{'os_returns_retained_on_zero' attribute only applies to pointer/reference-to-OSObject-pointer parameters}}

void write_into_out_parameter_ref(__attribute__((os_returns_retained)) S*& out) {}

typedef S* SPtr;

void write_into_out_parameter_typedef(__attribute__((os_returns_retained)) SPtr* out) {}

void write_into_out_parameter_invalid(__attribute__((os_returns_retained)) S* out) {} // expected-warning{{'os_returns_retained' attribute only applies to pointer/reference-to-OSObject-pointer parameters}}

void write_into_out_parameter_not_retained(__attribute__((os_returns_not_retained)) S **out) {}

void write_into_out_parameter_not_retained(__attribute__((os_returns_not_retained)) S volatile * const * const out) {}

void write_into_not_retainedout_parameter_invalid(__attribute__((os_returns_not_retained)) S* out) {} // expected-warning{{'os_returns_not_retained' attribute only applies to pointer/reference-to-OSObject-pointer parameters}}

void write_into_not_retainedout_parameter_too_many_pointers(__attribute__((os_returns_not_retained)) S*** out) {} // expected-warning{{'os_returns_not_retained' attribute only applies to pointer/reference-to-OSObject-pointer parameters}}
