// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fblocks -Wno-objc-root-class -analyze -analyzer-checker=core,deadcode,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s

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

int getValue();
void useValue(int v);

void castToBlockNoDeadStore() {
  int v = getValue(); // no-warning

  (void)(void(^)())[v]() { // This capture should count as a use, so no dead store warning above.
  };
}

void takesBlock(void(^block)());

void passToFunctionTakingBlockNoDeadStore() {
  int v = 7; // no-warning
  int x = 8; // no-warning
  takesBlock([&v, x]() {
    (void)v;
  });
}

void castToBlockAndInline() {
  int result = ((int(^)(int))[](int p) {
    return p;
  })(7);

  // FIXME: This should be TRUE. We're not handling lambda to block conversions
  // properly in ExprEngine::VisitBlockExpr.
  clang_analyzer_eval(result == 7); // expected-warning{{UNKNOWN}}
}

void castLambdaInLocalBlock() {
  // FIXME: This results in a spurious
  // "Address of stack-allocated block declared on line XX returned to caller" warning
  // because we're not handling lambda to block conversions properly in ExprEngine.
  auto lambda = []{ }; // expected-warning {{Address of stack-allocated block declared on line}}

  void(^block)() = lambda;
  (void)block;
}
