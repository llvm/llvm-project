// RUN: %clang_cc1 -fsyntax-only -verify %s

#define PLACE_IN_TCB(NAME) __attribute__ ((enforce_tcb(NAME)))
#define PLACE_IN_TCB_LEAF(NAME) __attribute__ ((enforce_tcb_leaf(NAME)))

void foo1 (void) PLACE_IN_TCB("bar");
void foo2 (void) PLACE_IN_TCB("bar");
void foo3 (void); // not in any TCB
void foo4 (void) PLACE_IN_TCB("bar2");
void foo5 (void) PLACE_IN_TCB_LEAF("bar");
void foo6 (void) PLACE_IN_TCB("bar2") PLACE_IN_TCB("bar");
void foo7 (void) PLACE_IN_TCB("bar3");
void foo8 (void) PLACE_IN_TCB("bar") PLACE_IN_TCB("bar2");
void foo9 (void);

void foo1(void) {
    foo2(); // OK - function in same TCB
    foo3(); // expected-warning {{calling 'foo3' is a violation of trusted computing base 'bar'}}
    foo4(); // expected-warning {{calling 'foo4' is a violation of trusted computing base 'bar'}}
    foo5(); // OK - in leaf node
    foo6(); // OK - in multiple TCBs, one of which is the same
    foo7(); // expected-warning {{calling 'foo7' is a violation of trusted computing base 'bar'}}
    (void) __builtin_clz(5); // OK - builtins are excluded
}

// Normal use without any attributes works
void foo3(void) {
    foo9(); // no-warning
}

void foo5(void) {
    // all calls should be okay, function in TCB leaf
    foo2(); // no-warning
    foo3(); // no-warning
    foo4(); // no-warning
}

void foo6(void) {
    foo1(); // expected-warning {{calling 'foo1' is a violation of trusted computing base 'bar2'}}
    foo4(); // expected-warning {{calling 'foo4' is a violation of trusted computing base 'bar'}}
    foo8(); // no-warning
    foo7(); // #1
    // expected-warning@#1 {{calling 'foo7' is a violation of trusted computing base 'bar2'}}
    // expected-warning@#1 {{calling 'foo7' is a violation of trusted computing base 'bar'}}
}

// Ensure that attribute merging works as expected across redeclarations.
void foo10(void) PLACE_IN_TCB("bar");
void foo10(void) PLACE_IN_TCB("bar2");
void foo10(void) PLACE_IN_TCB("bar3");
void foo10(void) {
  foo1(); // #2
    // expected-warning@#2 {{calling 'foo1' is a violation of trusted computing base 'bar2'}}
    // expected-warning@#2 {{calling 'foo1' is a violation of trusted computing base 'bar3'}}
  foo3(); // #3
    // expected-warning@#3 {{calling 'foo3' is a violation of trusted computing base 'bar'}}
    // expected-warning@#3 {{calling 'foo3' is a violation of trusted computing base 'bar2'}}
    // expected-warning@#3 {{calling 'foo3' is a violation of trusted computing base 'bar3'}}
  foo4(); // #4
    // expected-warning@#4 {{calling 'foo4' is a violation of trusted computing base 'bar'}}
    // expected-warning@#4 {{calling 'foo4' is a violation of trusted computing base 'bar3'}}
  foo7(); // #5
    // expected-warning@#5 {{calling 'foo7' is a violation of trusted computing base 'bar'}}
    // expected-warning@#5 {{calling 'foo7' is a violation of trusted computing base 'bar2'}}
}
