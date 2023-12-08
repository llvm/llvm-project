// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -Wno-objc-literal-conversion %s

void clang_analyzer_eval(bool);
@class NSString;

void sanity() {
  clang_analyzer_eval(@""); // expected-warning{{TRUE}}
  clang_analyzer_eval(@"abc"); // expected-warning{{TRUE}}
}

namespace rdar13773117 {
  NSString *const kConstantGlobalString = @"foo";
  NSString *globalString = @"bar";

  extern void invalidateGlobals();

  void testGlobals() {
    clang_analyzer_eval(kConstantGlobalString); // expected-warning{{TRUE}}
    clang_analyzer_eval(globalString); // expected-warning{{UNKNOWN}}

    globalString = @"baz";
    clang_analyzer_eval(globalString); // expected-warning{{TRUE}}

    invalidateGlobals();

    clang_analyzer_eval(kConstantGlobalString); // expected-warning{{TRUE}}
    clang_analyzer_eval(globalString); // expected-warning{{UNKNOWN}}
  }

  NSString *returnString(NSString *input = @"garply") {
    return input;
  }

  void testDefaultArg() {
    clang_analyzer_eval(returnString(@"")); // expected-warning{{TRUE}}
    clang_analyzer_eval(returnString(0)); // expected-warning{{FALSE}}
    clang_analyzer_eval(returnString()); // expected-warning{{TRUE}}
  }
}
