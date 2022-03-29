// RUN: %clang_cc1 -fsyntax-only -verify %s

int foo();

__attribute__((objc_root_class))
@interface AClass
- (void)bothTCBAndTCBLeafOnSeparateRedeclarations __attribute__((enforce_tcb("x"))); // expected-note{{conflicting attribute is here}}

- (void)bothTCBAndTCBLeafOnSeparateRedeclarationsOppositeOrder __attribute__((enforce_tcb_leaf("x"))); // expected-note{{conflicting attribute is here}}

- (void)bothTCBAndTCBLeafButDifferentIdentifiersOnSeparateRedeclarations __attribute__((enforce_tcb("x")));

- (void)bothTCBAndTCBLeafButDifferentIdentifiersOnSeparateRedeclarationsOppositeOrder __attribute__((enforce_tcb_leaf("x")));

- (void)onInterfaceOnly __attribute__((enforce_tcb("test")));
@end

@interface AClass (NoImplementation)
- (void)noArguments __attribute__((enforce_tcb)); // expected-error{{'enforce_tcb' attribute takes one argument}}

- (void)tooManyArguments __attribute__((enforce_tcb("test", 12))); // expected-error{{'enforce_tcb' attribute takes one argument}}

- (void)wrongArgumentType __attribute__((enforce_tcb(12))); // expected-error{{'enforce_tcb' attribute requires a string}}

- (void)noArgumentsLeaf __attribute__((enforce_tcb_leaf)); // expected-error{{'enforce_tcb_leaf' attribute takes one argument}}

- (void)tooManyArgumentsLeaf __attribute__((enforce_tcb_leaf("test", 12))); // expected-error{{'enforce_tcb_leaf' attribute takes one argument}}

- (void)wrongArgumentTypeLeaf __attribute__((enforce_tcb_leaf(12))); // expected-error{{'enforce_tcb_leaf' attribute requires a string}}
@end

@implementation AClass
- (void)onInterfaceOnly {
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'test'}}
}

- (void)bothTCBAndTCBLeaf
    __attribute__((enforce_tcb("x")))
    __attribute__((enforce_tcb_leaf("x"))) // expected-error{{attributes 'enforce_tcb_leaf("x")' and 'enforce_tcb("x")' are mutually exclusive}}
{
  foo(); // no-warning
}

- (void)bothTCBAndTCBLeafOnSeparateRedeclarations
    __attribute__((enforce_tcb_leaf("x"))) // expected-error{{attributes 'enforce_tcb_leaf("x")' and 'enforce_tcb("x")' are mutually exclusive}}
{
  // Error recovery: no need to emit a warning when we didn't
  // figure out our attributes to begin with.
  foo(); // no-warning
}

- (void)bothTCBAndTCBLeafOppositeOrder
    __attribute__((enforce_tcb_leaf("x")))
    __attribute__((enforce_tcb("x"))) // expected-error{{attributes 'enforce_tcb("x")' and 'enforce_tcb_leaf("x")' are mutually exclusive}}
{
  foo(); // no-warning
}

- (void)bothTCBAndTCBLeafOnSeparateRedeclarationsOppositeOrder
    __attribute__((enforce_tcb("x"))) // expected-error{{attributes 'enforce_tcb("x")' and 'enforce_tcb_leaf("x")' are mutually exclusive}}
{
  foo(); // no-warning
}

- (void)bothTCBAndTCBLeafButDifferentIdentifiers
    __attribute__((enforce_tcb("x")))
    __attribute__((enforce_tcb_leaf("y"))) // no-error
{
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'x'}}
}

- (void)bothTCBAndTCBLeafButDifferentIdentifiersOppositeOrder
    __attribute__((enforce_tcb_leaf("x")))
    __attribute__((enforce_tcb("y"))) // no-error
{
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'y'}}
}

- (void)bothTCBAndTCBLeafButDifferentIdentifiersOnSeparateRedeclarations
    __attribute__((enforce_tcb_leaf("y"))) // no-error
{
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'x'}}
}

- (void)bothTCBAndTCBLeafButDifferentIdentifiersOnSeparateRedeclarationsOppositeOrder
    __attribute__((enforce_tcb("y"))) {
  foo(); // expected-warning{{calling 'foo' is a violation of trusted computing base 'y'}}
}

- (void)errorRecoveryOverIndividualTCBs
    __attribute__((enforce_tcb("y")))
    __attribute__((enforce_tcb("x")))
    __attribute__((enforce_tcb_leaf("x"))) // expected-error{{attributes 'enforce_tcb_leaf("x")' and 'enforce_tcb("x")' are mutually exclusive}}
{
  // FIXME: Ideally this should warn. The conflict between attributes
  // for TCB "x" shouldn't affect the warning about TCB "y".
  foo(); // no-warning
}

@end
