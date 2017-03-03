// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,debug.ExprInspection -analyzer-config ipa=none -fblocks -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,debug.ExprInspection -analyzer-config ipa=none -fblocks -verify -x c++ %s

void clang_analyzer_eval(int);

void testInvalidation() {
  __block int i = 0;
  ^{
    ++i;
  }();

  // Under inlining, we will know that i == 1.
  clang_analyzer_eval(i == 0); // expected-warning{{UNKNOWN}}
}


const int globalConstant = 1;
void testCapturedConstants() {
  const int localConstant = 2;
  static const int staticConstant = 3;

  ^{
    clang_analyzer_eval(globalConstant == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(localConstant == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(staticConstant == 3); // expected-warning{{TRUE}}
  }();
}

typedef const int constInt;
constInt anotherGlobalConstant = 1;
void testCapturedConstantsTypedef() {
  constInt localConstant = 2;
  static constInt staticConstant = 3;

  ^{
    clang_analyzer_eval(anotherGlobalConstant == 1); // expected-warning{{TRUE}}
    clang_analyzer_eval(localConstant == 2); // expected-warning{{TRUE}}
    clang_analyzer_eval(staticConstant == 3); // expected-warning{{TRUE}}
  }();
}
