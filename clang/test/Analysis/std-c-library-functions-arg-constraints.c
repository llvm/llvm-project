// Check the basic reporting/warning and the application of constraints.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -verify=report

// Check the bugpath related to the reports.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctionArgs \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -triple x86_64-unknown-linux-gnu \
// RUN:   -analyzer-output=text \
// RUN:   -verify=bugpath

void clang_analyzer_eval(int);

int glob;

#define EOF -1

int isalnum(int);

void test_alnum_concrete(int v) {
  int ret = isalnum(256); // \
  // report-warning{{Function argument constraint is not satisfied}} \
  // bugpath-warning{{Function argument constraint is not satisfied}} \
  // bugpath-note{{Function argument constraint is not satisfied}}
  (void)ret;
}

void test_alnum_symbolic(int x) {
  int ret = isalnum(x);
  (void)ret;

  clang_analyzer_eval(EOF <= x && x <= 255); // \
  // report-warning{{TRUE}} \
  // bugpath-warning{{TRUE}} \
  // bugpath-note{{TRUE}} \
  // bugpath-note{{Left side of '&&' is true}} \
  // bugpath-note{{'x' is <= 255}}

}

void test_alnum_symbolic2(int x) {
  if (x > 255) { // \
    // bugpath-note{{Assuming 'x' is > 255}} \
    // bugpath-note{{Taking true branch}}

    int ret = isalnum(x); // \
    // report-warning{{Function argument constraint is not satisfied}} \
    // bugpath-warning{{Function argument constraint is not satisfied}} \
    // bugpath-note{{Function argument constraint is not satisfied}}

    (void)ret;
  }
}
