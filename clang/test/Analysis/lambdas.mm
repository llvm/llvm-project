// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -Wno-objc-root-class -analyzer-checker=core,deadcode,debug.ExprInspection -analyzer-config inline-lambdas=true -verify %s

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

// Make sure to properly handle super-calls when a block captures
// a local variable named 'self'.
- (void)callMethodOnSuperInCXXLambdaWithRedefinedSelf; {
  /*__weak*/ Sub *weakSelf = self;
  // Implicit capture. (Sema outlaws explicit capture of a redefined self
  // and a call to super [which uses the original self]).
  [=]() {
    Sub *self = weakSelf;
    [=]() {
      [super superMethod];
    }();
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

  clang_analyzer_eval(result == 7); // expected-warning{{TRUE}}
}

void castToBlockWithCaptureAndInline() {
  int y = 7;

  auto lambda = [y]{ return y; };
  int(^block)() = lambda;

  int result = block();
  clang_analyzer_eval(result == 7); // expected-warning{{TRUE}}
}

void castMutableLambdaToBlock() {
  int x = 0;

  auto lambda = [x]() mutable {
    x = x + 1;
    return x;
   };

  // The block should copy the lambda before capturing.
  int(^block)() = lambda;

  int r1 = block();
  clang_analyzer_eval(r1 == 1); // expected-warning{{TRUE}}

  int r2 = block();
  clang_analyzer_eval(r2 == 2); // expected-warning{{TRUE}}

  // Because block copied the lambda, r3 should be 1.
  int r3 = lambda();
  clang_analyzer_eval(r3 == 1); // expected-warning{{TRUE}}

  // Aliasing the block shouldn't copy the lambda.
  int(^blockAlias)() = block;

  int r4 = blockAlias();
  clang_analyzer_eval(r4 == 3); // expected-warning{{TRUE}}

  int r5 = block();
  clang_analyzer_eval(r5 == 4); // expected-warning{{TRUE}}

  // Another copy of lambda
  int(^blockSecondCopy)() = lambda;
  int r6 = blockSecondCopy();
  clang_analyzer_eval(r6 == 2); // expected-warning{{TRUE}}
}

void castLambdaInLocalBlock() {
  // Make sure we don't emit a spurious diagnostic about the address of a block
  // escaping in the implicit conversion operator method for lambda-to-block
  // conversions.
  auto lambda = []{ }; // no-warning

  void(^block)() = lambda;
  (void)block;
}
