// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify -DEXPECT_NO_DIAGNOSTICS %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify=conditional %s \
// RUN:    -analyzer-config ignore-bison-generated-files=false

#ifdef EXPECT_NO_DIAGNOSTICS
// expected-no-diagnostics
#endif

/* A Bison parser, made by GNU Bison 1.875.  */

void clang_analyzer_warnIfReached(void);
void foo(void) {
  clang_analyzer_warnIfReached(); // conditional-warning {{REACHABLE}}
}
