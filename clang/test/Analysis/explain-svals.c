// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -analyzer-checker=core.builtin,debug.ExprInspection,unix.cstring -verify %s

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
