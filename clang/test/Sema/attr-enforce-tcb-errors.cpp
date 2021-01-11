// RUN: %clang_cc1 -fsyntax-only -verify %s

[[clang::enforce_tcb("oops")]] int wrong_subject_type; // expected-warning{{'enforce_tcb' attribute only applies to functions}}

void no_arguments() __attribute__((enforce_tcb)); // expected-error{{'enforce_tcb' attribute takes one argument}}

void too_many_arguments() __attribute__((enforce_tcb("test", 12))); // expected-error{{'enforce_tcb' attribute takes one argument}}

void wrong_argument_type() __attribute__((enforce_tcb(12))); // expected-error{{'enforce_tcb' attribute requires a string}}

[[clang::enforce_tcb_leaf("oops")]] int wrong_subject_type_leaf; // expected-warning{{'enforce_tcb_leaf' attribute only applies to functions}}

void no_arguments_leaf() __attribute__((enforce_tcb_leaf)); // expected-error{{'enforce_tcb_leaf' attribute takes one argument}}

void too_many_arguments_leaf() __attribute__((enforce_tcb_leaf("test", 12))); // expected-error{{'enforce_tcb_leaf' attribute takes one argument}}
void wrong_argument_type_leaf() __attribute__((enforce_tcb_leaf(12))); // expected-error{{'enforce_tcb_leaf' attribute requires a string}}

void foo();

__attribute__((enforce_tcb("x")))
__attribute__((enforce_tcb_leaf("x"))) // expected-error{{attributes 'enforce_tcb_leaf("x")' and 'enforce_tcb("x")' are mutually exclusive}}
void both_tcb_and_tcb_leaf() {
  foo(); // no-warning
}

__attribute__((enforce_tcb_leaf("x"))) // expected-note{{conflicting attribute is here}}
void both_tcb_and_tcb_leaf_on_separate_redeclarations();
__attribute__((enforce_tcb("x"))) // expected-error{{attributes 'enforce_tcb("x")' and 'enforce_tcb_leaf("x")' are mutually exclusive}}
void both_tcb_and_tcb_leaf_on_separate_redeclarations() {
  // Error recovery: no need to emit a warning when we didn't
  // figure out our attributes to begin with.
  foo(); // no-warning
}

__attribute__((enforce_tcb_leaf("x")))
__attribute__((enforce_tcb("x"))) // expected-error{{attributes 'enforce_tcb("x")' and 'enforce_tcb_leaf("x")' are mutually exclusive}}
void both_tcb_and_tcb_leaf_opposite_order() {
  foo(); // no-warning
}

__attribute__((enforce_tcb("x"))) // expected-note{{conflicting attribute is here}}
void both_tcb_and_tcb_leaf_on_separate_redeclarations_opposite_order();
__attribute__((enforce_tcb_leaf("x"))) // expected-error{{attributes 'enforce_tcb_leaf("x")' and 'enforce_tcb("x")' are mutually exclusive}}
void both_tcb_and_tcb_leaf_on_separate_redeclarations_opposite_order() {
  foo(); // no-warning
}

__attribute__((enforce_tcb("x")))
__attribute__((enforce_tcb_leaf("y"))) // no-error
void both_tcb_and_tcb_leaf_but_different_identifiers() {
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'x'}}
}
__attribute__((enforce_tcb_leaf("x")))
__attribute__((enforce_tcb("y"))) // no-error
void both_tcb_and_tcb_leaf_but_different_identifiers_opposite_order() {
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'y'}}
}

__attribute__((enforce_tcb("x")))
void both_tcb_and_tcb_leaf_but_different_identifiers_on_separate_redeclarations();
__attribute__((enforce_tcb_leaf("y"))) // no-error
void both_tcb_and_tcb_leaf_but_different_identifiers_on_separate_redeclarations() {
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'x'}}
}

__attribute__((enforce_tcb_leaf("x")))
void both_tcb_and_tcb_leaf_but_different_identifiers_on_separate_redeclarations_opposite_order();
__attribute__((enforce_tcb("y")))
void both_tcb_and_tcb_leaf_but_different_identifiers_on_separate_redeclarations_opposite_order() {
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'y'}}
}

__attribute__((enforce_tcb("y")))
__attribute__((enforce_tcb("x")))
__attribute__((enforce_tcb_leaf("x"))) // expected-error{{attributes 'enforce_tcb_leaf("x")' and 'enforce_tcb("x")' are mutually exclusive}}
void error_recovery_over_individual_tcbs() {
  // FIXME: Ideally this should warn. The conflict between attributes
  // for TCB "x" shouldn't affect the warning about TCB "y".
  foo(); // no-warning
}
