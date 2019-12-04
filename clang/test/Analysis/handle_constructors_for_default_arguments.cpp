// RUN: %clang_cc1 -fsyntax-only -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection %s -verify

// These test cases demonstrate lack of Static Analyzer features.
// The FIXME: tags indicate where we expect different output.

// Handle constructors for default arguments.
// Default arguments in C++ are recomputed at every call,
// and are therefore local, and not static, variables.
void clang_analyzer_eval(bool);
void clang_analyzer_warnIfReached();

struct init_with_list {
  int a;
  init_with_list() : a(1) {}
};

struct init_in_body {
  int a;
  init_in_body() { a = 1; }
};

struct init_default_member {
  int a = 1;
};

struct basic_struct {
  int a;
};

// Top-level analyzed functions.
void top_f(init_with_list l = init_with_list()) {
  // We expect that the analyzer doesn't assume anything about the parameter.
  clang_analyzer_eval(l.a == 1); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}

void top_g(init_in_body l = init_in_body()) {
  // We expect that the analyzer doesn't assume anything about the parameter.
  clang_analyzer_eval(l.a == 1); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}

void top_h(init_default_member l = init_default_member()) {
  // We expect that the analyzer doesn't assume anything about the parameter.
  clang_analyzer_eval(l.a == 1); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}

// Not-top-level analyzed functions.
int called_f(init_with_list l = init_with_list()) {
  // We expect that the analyzer assumes the default value
  // when called from test2().
  return l.a;
}

int called_g(init_in_body l = init_in_body()) {
  // We expect that the analyzer assumes the default value
  // when called from test3().
  return l.a;
}

int called_h(init_default_member l = init_default_member()) {
  // We expect that the analyzer assumes the default value
  // when called from test4().
  return l.a;
}

int called_i(const init_with_list &l = init_with_list()){
  // We expect that the analyzer assumes the default value
  // when called from test5().
  return l.a;
}

int called_j(init_with_list &&l = init_with_list()){
  // We expect that the analyzer assumes the default value
  // when called from test6().
  return l.a;
}

int plain_parameter_passing(basic_struct l) {
  return l.a;
}

void test1() {
  basic_struct b;
  b.a = 1;
  clang_analyzer_eval(plain_parameter_passing(b) == 1); //expected-warning {{TRUE}}
}

void test2() {
  // We expect that the analyzer assumes the default value.
  // FIXME: Should be TRUE.
  clang_analyzer_eval(called_f() == 1); //expected-warning {{TRUE}} expected-warning {{FALSE}}
}

void test3() {
  // We expect that the analyzer assumes the default value.
  // FIXME: Should be TRUE.
  clang_analyzer_eval(called_g() == 1); //expected-warning {{TRUE}} expected-warning {{FALSE}}
}

void test4() {
  // We expect that the analyzer assumes the default value.
  // FIXME: Should be TRUE.
  clang_analyzer_eval(called_h() == 1); //expected-warning {{TRUE}} expected-warning {{FALSE}}
}

void test5() {
  //We expect that the analyzer assumes the default value.
  // FIXME: Should be TRUE.
  clang_analyzer_eval(called_i() == 1); //expected-warning {{TRUE}} expected-warning {{FALSE}}
}

void test6() {
  // We expect that the analyzer assumes the default value.
  // FIXME: Should be TRUE.
  clang_analyzer_eval(called_j() == 1); //expected-warning {{TRUE}} expected-warning {{FALSE}}
}
