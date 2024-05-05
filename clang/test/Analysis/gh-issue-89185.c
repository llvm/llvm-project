// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_dump(char);
void clang_analyzer_dump_ptr(char*);

// https://github.com/llvm/llvm-project/issues/89185
void binding_to_label_loc() {
  char *b = &&MyLabel;
MyLabel:
  *b = 0; // expected-warning {{Dereference of the address of a label}}
  clang_analyzer_dump_ptr(b);
  clang_analyzer_dump(*b);
}
