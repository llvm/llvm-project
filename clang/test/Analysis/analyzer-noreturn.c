// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached(void);
void analyzer_stop(void) __attribute__((analyzer_noreturn));
void real_stop(void) __attribute__((noreturn));

// The static analyzer treats both 'analyzer_noreturn' and 'noreturn' calls
// as sinks.
void analyzer(void) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  analyzer_stop();
  clang_analyzer_warnIfReached(); // no-warning
}

void real(void) {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
  real_stop();
  clang_analyzer_warnIfReached(); // no-warning
}
