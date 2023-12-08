// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -triple x86_64-pc-windows-msvc19.11.0 -fms-extensions -verify %s

void clang_analyzer_warnIfReached();
int filter();

void try_except_leave() {
  __try {
    __leave;                        // no-crash
    clang_analyzer_warnIfReached(); // no-warning
  } __except (filter()) {
  }
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
