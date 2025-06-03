// RUN: %clang_analyze_cc1 -std=c++11 -verify %s \
// RUN:   -triple=x86_64-unknown-linux-gnu \
// RUN:   -analyzer-checker=core,security.ArrayBound,debug.ExprInspection

void clang_analyzer_eval(bool);
void clang_analyzer_value(int);
void clang_analyzer_dump(int);

// From: https://github.com/llvm/llvm-project/issues/100762
extern int arrOf10[10];
void using_builtin(int x) {
  __builtin_assume(x > 101); // CallExpr
  arrOf10[x] = 404; // expected-warning {{Out of bound access to memory}}
}

void using_assume_attr(int ax) {
  [[assume(ax > 100)]]; // NullStmt with an "assume" attribute.
  arrOf10[ax] = 405; // expected-warning {{Out of bound access to memory}}
}

void using_many_assume_attr(int yx) {
  [[assume(yx > 104), assume(yx > 200), assume(yx < 300)]]; // NullStmt with an attribute
  arrOf10[yx] = 406; // expected-warning{{Out of bound access to memory}}
}

int using_assume_attr_has_no_sideeffects(int y) {
  int orig_y = y;
  clang_analyzer_value(y);      // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_value(orig_y); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_dump(y);       // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}
  clang_analyzer_dump(orig_y);  // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}

  // We should not apply sideeffects of the argument of [[assume(...)]].
  // "y" should not get incremented;
  [[assume(++y == 43)]]; // expected-warning {{assumption is ignored because it contains (potential) side-effects}}

  clang_analyzer_dump(y);       // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}
  clang_analyzer_dump(orig_y);  // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}
  clang_analyzer_value(y);      // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_value(orig_y); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_eval(y == orig_y); // expected-warning {{TRUE}} Good.

  return y;
}

int using_builtin_assume_has_no_sideeffects(int y) {
  int orig_y = y;
  clang_analyzer_value(y);      // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_value(orig_y); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_dump(y);       // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}
  clang_analyzer_dump(orig_y);  // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}

  // We should not apply sideeffects of the argument of __builtin_assume(...)
  // "u" should not get incremented;
  __builtin_assume(++y == 43); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}

  clang_analyzer_dump(y);       // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}
  clang_analyzer_dump(orig_y);  // expected-warning-re {{{{^}}reg_${{[0-9]+}}<int y> [debug.ExprInspection]{{$}}}}
  clang_analyzer_value(y);      // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_value(orig_y); // expected-warning {{32s:{ [-2147483648, 2147483647] }}}
  clang_analyzer_eval(y == orig_y); // expected-warning {{TRUE}} Good.

  return y;
}
