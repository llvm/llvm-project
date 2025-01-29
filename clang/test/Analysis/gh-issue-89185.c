// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-output text -verify %s 

void clang_analyzer_warnIfReached(void);

// https://github.com/llvm/llvm-project/issues/89185
void binding_to_label_loc() {
  char *b = &&MyLabel; // expected-note {{'b' initialized here}}
MyLabel:
  *b = 0;
  // expected-warning@-1 {{Dereference of the address of a label}}
  // expected-note@-2    {{Dereference of the address of a label}}
  clang_analyzer_warnIfReached(); // no-warning: Unreachable due to fatal error.
}
