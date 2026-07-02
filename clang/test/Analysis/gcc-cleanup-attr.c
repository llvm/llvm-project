// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_dump(int);

void defined_cleanup(int *p) {
  clang_analyzer_dump(*p); // expected-warning {{42}}
                           // expected-warning@-1 {{43}}
}

void vardecl_defined() {
  int x __attribute__((cleanup(defined_cleanup))) = 42;
  int y __attribute__((cleanup(defined_cleanup))) = 43;
}
