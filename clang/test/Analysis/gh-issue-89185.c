// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_dump(char);
void clang_analyzer_dump_ptr(char*);

// https://github.com/llvm/llvm-project/issues/89185
void binding_to_label_loc() {
  char *b = &&MyLabel;
MyLabel:
  *b = 0; // no-crash
  clang_analyzer_dump_ptr(b); // expected-warning {{&&MyLabel}}
  clang_analyzer_dump(*b); // expected-warning {{Unknown}}
  // FIXME: We should never reach here, as storing to a label is invalid.
}
