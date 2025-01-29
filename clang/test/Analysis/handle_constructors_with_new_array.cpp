// RUN: %clang_cc1 -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection %s -verify

// These test cases demonstrate lack of Static Analyzer features.
// The FIXME: tags indicate where we expect different output.

// Handle constructors within new[].

// When an array of objects is allocated using the operator new[],
// constructors for all elements of the array are called.
// We should model (potentially some of) such evaluations,
// and the same applies for destructors called from operator delete[].

void clang_analyzer_eval(bool);

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

void test_automatic() {

  init_with_list a1;
  init_in_body a2;
  init_default_member a3;

  clang_analyzer_eval(a1.a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a2.a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a3.a == 1); // expected-warning {{TRUE}}
}

void test_dynamic() {

  auto *a1 = new init_with_list;
  auto *a2 = new init_in_body;
  auto *a3 = new init_default_member;

  clang_analyzer_eval(a1->a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a2->a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a3->a == 1); // expected-warning {{TRUE}}

  delete a1;
  delete a2;
  delete a3;
}

void test_automatic_aggregate() {

  init_with_list a1[1];
  init_in_body a2[1];
  init_default_member a3[1];

  clang_analyzer_eval(a1[0].a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a2[0].a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a3[0].a == 1); // expected-warning {{TRUE}}
}

void test_dynamic_aggregate() {

  auto *a1 = new init_with_list[1];
  auto *a2 = new init_in_body[1];
  auto *a3 = new init_default_member[1];

  clang_analyzer_eval(a1[0].a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a2[0].a == 1); // expected-warning {{TRUE}}
  clang_analyzer_eval(a3[0].a == 1); // expected-warning {{TRUE}}

  delete[] a1;
  delete[] a2;
  delete[] a3;
}
