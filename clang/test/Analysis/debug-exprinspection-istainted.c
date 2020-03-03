// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-checker=alpha.security.taint

int scanf(const char *restrict format, ...);
void clang_analyzer_isTainted(char);
void clang_analyzer_isTainted_any_suffix(char);
void clang_analyzer_isTainted_many_arguments(char, int, int);

void foo() {
  char buf[32] = "";
  clang_analyzer_isTainted(buf[0]);            // expected-warning {{NO}}
  clang_analyzer_isTainted_any_suffix(buf[0]); // expected-warning {{NO}}
  scanf("%s", buf);
  clang_analyzer_isTainted(buf[0]);            // expected-warning {{YES}}
  clang_analyzer_isTainted_any_suffix(buf[0]); // expected-warning {{YES}}

  int tainted_value = buf[0]; // no-warning
}

void exactly_one_argument_required() {
  char buf[32] = "";
  scanf("%s", buf);
  clang_analyzer_isTainted_many_arguments(buf[0], 42, 42);
  // expected-warning@-1 {{clang_analyzer_isTainted() requires exactly one argument}}
}
