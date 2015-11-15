// RUN: %clang_cc1 -std=c++11 -fsyntax-only -Wno-objc-root-class -analyze -analyzer-checker=core,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s

int clang_analyzer_eval(int);

@interface Super
- (void)superMethod;
@end

@interface Sub : Super {
  int _ivar1;
  int _ivar2;
}
@end


@implementation Sub
- (void)callMethodOnSuperInCXXLambda; {
  // Explicit capture.
  [self]() {
    [super superMethod];
  }();

  // Implicit capture.
  [=]() {
    [super superMethod];
  }();
}

- (void)swapIvars {
  int tmp = _ivar1;
  _ivar1 = _ivar2;
  _ivar2 = tmp;
}

- (void)callMethodOnSelfInCXXLambda; {
  _ivar1 = 7;
  _ivar2 = 8;
  [self]() {
    [self swapIvars];
  }();

  clang_analyzer_eval(_ivar1 == 8); // expected-warning{{TRUE}}
  clang_analyzer_eval(_ivar2 == 7); // expected-warning{{TRUE}}
}

@end
