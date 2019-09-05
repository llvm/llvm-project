// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: [ \
// RUN:     {key: bugprone-argument-comment.IgnoreSingleArgument, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentBoolLiterals, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentIntegerLiterals, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentFloatLiterals, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentUserDefinedLiterals, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentStringLiterals, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentNullPtrs, value: 1}, \
// RUN:     {key: bugprone-argument-comment.CommentCharacterLiterals, value: 1}]}" --

struct A {
  void foo(bool abc);
  void foo(bool abc, bool cde);
  void foo(const char *, bool abc);
  void foo(int iabc);
  void foo(float fabc);
  void foo(double dabc);
  void foo(const char *strabc);
  void fooW(const wchar_t *wstrabc);
  void fooPtr(A *ptrabc);
  void foo(char chabc);
};

#define FOO 1

void g(int a);
void h(double b);
void i(const char *c);

double operator"" _km(long double);

void test() {
  A a;

  a.foo(true);

  a.foo(false);

  a.foo(true, false);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:15: warning: argument comment missing for literal argument 'cde' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*abc=*/true, /*cde=*/false);

  a.foo(false, true);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:16: warning: argument comment missing for literal argument 'cde' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*abc=*/false, /*cde=*/true);

  a.foo(/*abc=*/false, true);
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: argument comment missing for literal argument 'cde' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*abc=*/false, /*cde=*/true);

  a.foo(false, /*cde=*/true);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*abc=*/false, /*cde=*/true);

  bool val1 = true;
  bool val2 = false;
  a.foo(val1, val2);

  a.foo("", true);
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo("", /*abc=*/true);

  a.foo(0);

  a.foo(1.0f);

  a.foo(1.0);

  int val3 = 10;
  a.foo(val3);

  float val4 = 10.0;
  a.foo(val4);

  double val5 = 10.0;
  a.foo(val5);

  a.foo("Hello World");

  a.fooW(L"Hello World");

  a.fooPtr(nullptr);

  a.foo(402.0_km);

  a.foo('A');

  g(FOO);

  h(1.0f);

  i(__FILE__);

  g((1));
}

void f(bool _with_underscores_);
void ignores_underscores() {
  f(false);

  f(true);
}
