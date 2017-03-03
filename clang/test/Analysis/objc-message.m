// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-store=region -verify -Wno-objc-root-class %s

extern void clang_analyzer_warnIfReached();
void clang_analyzer_eval(int);

@interface SomeClass
-(id)someMethodWithReturn;
-(void)someMethod;
@end

void consistencyOfReturnWithNilReceiver(SomeClass *o) {
  id result = [o someMethodWithReturn];
  if (result) {
    if (!o) {
      // It is impossible for both o to be nil and result to be non-nil,
      // so this should not be reached.
      clang_analyzer_warnIfReached(); // no-warning
    }
  }
}

void maybeNilReceiverIsNotNilAfterMessage(SomeClass *o) {
  [o someMethod];

  // We intentionally drop the nil flow (losing coverage) after a method
  // call when the receiver may be nil in order to avoid inconsistencies of
  // the kind tested for in consistencyOfReturnWithNilReceiver().
  clang_analyzer_eval(o != 0); // expected-warning{{TRUE}}
}

void nilReceiverIsStillNilAfterMessage(SomeClass *o) {
  if (o == 0) {
    id result = [o someMethodWithReturn];

    // Both the receiver and the result should be nil after a message
    // sent to a nil receiver returning a value of type id.
    clang_analyzer_eval(o == 0); // expected-warning{{TRUE}}
    clang_analyzer_eval(result == 0); // expected-warning{{TRUE}}
  }
}
