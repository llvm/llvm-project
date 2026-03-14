// RUN: %clang_cc1 -fsyntax-only -verify %s

#define PLACE_IN_TCB(NAME) [[clang::enforce_tcb(NAME)]]
#define PLACE_IN_TCB_LEAF(NAME) [[clang::enforce_tcb_leaf(NAME)]]

PLACE_IN_TCB("foo") void in_tcb_foo();
void not_in_tcb();

// Test behavior on classes and methods.
class C {
  void bar();

  PLACE_IN_TCB("foo")
  void foo() {
    // TODO: Figure out if we want to support methods at all.
    // Does it even make sense to isolate individual methods into a TCB?
    // Maybe a per-class attribute would make more sense?
    bar(); // expected-warning{{calling 'bar' is a violation of trusted computing base 'foo'}}
  }
};

// Test behavior on templates.
template <typename Ty>
PLACE_IN_TCB("foo")
void foo_never_instantiated() {
  not_in_tcb(); // expected-warning{{calling 'not_in_tcb' is a violation of trusted computing base 'foo'}}
  in_tcb_foo(); // no-warning
}

template <typename Ty>
PLACE_IN_TCB("foo")
void foo_specialized();

template<>
void foo_specialized<int>() {
  not_in_tcb(); // expected-warning{{calling 'not_in_tcb' is a violation of trusted computing base 'foo'}}
  in_tcb_foo(); // no-warning
}

PLACE_IN_TCB("foo")
void call_template_good() {
  foo_specialized<int>(); // no-warning
}
PLACE_IN_TCB("bar")
void call_template_bad() {
  foo_specialized<int>(); // expected-warning{{calling 'foo_specialized<int>' is a violation of trusted computing base 'bar'}}
}

template<typename Ty>
void foo_specialization_in_tcb();

template<>
PLACE_IN_TCB("foo")
void foo_specialization_in_tcb<int>() {
  not_in_tcb(); //expected-warning{{calling 'not_in_tcb' is a violation of trusted computing base 'foo'}}
  in_tcb_foo(); // no-warning
}

template<>
void foo_specialization_in_tcb<double>() {
  not_in_tcb(); // no-warning
  in_tcb_foo(); // no-warning
}

PLACE_IN_TCB("foo")
void call_specialization_in_tcb() {
  foo_specialization_in_tcb<int>(); // no-warning
  foo_specialization_in_tcb<long>(); // expected-warning{{calling 'foo_specialization_in_tcb<long>' is a violation of trusted computing base 'foo'}}
  foo_specialization_in_tcb<double>(); // expected-warning{{'foo_specialization_in_tcb<double>' is a violation of trusted computing base 'foo'}}
}
