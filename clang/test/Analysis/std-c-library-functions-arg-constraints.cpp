// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.unix.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.StdCLibraryFunctionsTester \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux \
// RUN:   -verify

void clang_analyzer_eval(int);

int __defaultparam(void *, int y = 3);

void test_arg_constraint_on_fun_with_default_param() {
  __defaultparam(nullptr); // \
  // expected-warning{{The 1st argument to '__defaultparam' is NULL but should not be NULL}}
}
