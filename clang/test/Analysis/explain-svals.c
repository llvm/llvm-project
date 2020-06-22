// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -verify %s \
// RUN:   -analyzer-checker=core.builtin \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=unix.cstring \
// RUN:   -analyzer-config display-checker-name=false

struct S {
  int z;
};

void clang_analyzer_explain_int(int);
void clang_analyzer_explain_voidp(void *);
void clang_analyzer_explain_S(struct S);

int glob;

void test_1(int param, void *ptr) {
  clang_analyzer_explain_voidp(&glob); // expected-warning-re{{{{^pointer to global variable 'glob'$}}}}
  clang_analyzer_explain_int(param);   // expected-warning-re{{{{^argument 'param'$}}}}
  clang_analyzer_explain_voidp(ptr);   // expected-warning-re{{{{^argument 'ptr'$}}}}
  if (param == 42)
    clang_analyzer_explain_int(param); // expected-warning-re{{{{^signed 32-bit integer '42'$}}}}
}

void test_2(struct S s) {
  clang_analyzer_explain_S(s);      //expected-warning-re{{{{^lazily frozen compound value of parameter 's'$}}}}
  clang_analyzer_explain_voidp(&s); // expected-warning-re{{{{^pointer to parameter 's'$}}}}
  clang_analyzer_explain_int(s.z);  // expected-warning-re{{{{^initial value of field 'z' of parameter 's'$}}}}
}

void test_3(int param) {
  clang_analyzer_explain_voidp(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
}

void test_non_top_level(int param) {
  clang_analyzer_explain_voidp(&param); // expected-warning-re{{{{^pointer to parameter 'param'$}}}}
}

void test_4(int n) {
  test_non_top_level(n);
}
