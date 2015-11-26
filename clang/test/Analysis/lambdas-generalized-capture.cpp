// RUN: %clang_cc1 -std=c++14 -fsyntax-only -analyze -analyzer-checker=core,deadcode,debug.ExprInspection -verify %s

int clang_analyzer_eval(int);

void generalizedCapture() {
  int v = 7;
  auto lambda = [x=v]() {
    return x;
  };

  int result = lambda();
  clang_analyzer_eval(result == 7); // expected-warning {{TRUE}}
}

void sideEffectsInGeneralizedCapture() {
  int v = 7;
  auto lambda = [x=v++]() {
    return x;
  };
  clang_analyzer_eval(v == 8); // expected-warning {{TRUE}}

  int r1 = lambda();
  int r2 = lambda();
  clang_analyzer_eval(r1 == 7); // expected-warning {{TRUE}}
  clang_analyzer_eval(r2 == 7); // expected-warning {{TRUE}}
  clang_analyzer_eval(v == 8); // expected-warning {{TRUE}}
}

int addOne(int p) {
 return p + 1;
}

void inliningInGeneralizedCapture() {
  int v = 7;
  auto lambda = [x=addOne(v)]() {
    return x;
  };

  int result = lambda();
  clang_analyzer_eval(result == 8); // expected-warning {{TRUE}}
}

void caseSplitInGeneralizedCapture(bool p) {
  auto lambda = [x=(p ? 1 : 2)]() {
    return x;
  };

  int result = lambda();
  clang_analyzer_eval(result == 1); // expected-warning {{FALSE}} expected-warning {{TRUE}}
}
