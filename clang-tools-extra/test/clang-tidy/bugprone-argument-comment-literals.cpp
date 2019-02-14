// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: [{key: CommentBoolLiterals, value: 1},{key: CommentIntegerLiterals, value: 1}, {key: CommentFloatLiterals, value: 1}, {key: CommentUserDefinedLiterals, value: 1}, {key: CommentStringLiterals, value: 1}, {key: CommentNullPtrs, value: 1}, {key: CommentCharacterLiterals, value: 1}]}" --

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
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*abc=*/true);

  a.foo(false);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*abc=*/false);

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
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'iabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*iabc=*/0);

  a.foo(1.0f);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'fabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*fabc=*/1.0f);

  a.foo(1.0);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'dabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*dabc=*/1.0);

  int val3 = 10;
  a.foo(val3);

  float val4 = 10.0;
  a.foo(val4);

  double val5 = 10.0;
  a.foo(val5);

  a.foo("Hello World");
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'strabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*strabc=*/"Hello World");
  //
  a.fooW(L"Hello World");
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: argument comment missing for literal argument 'wstrabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.fooW(/*wstrabc=*/L"Hello World");

  a.fooPtr(nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: argument comment missing for literal argument 'ptrabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.fooPtr(/*ptrabc=*/nullptr);

  a.foo(402.0_km);
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'dabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*dabc=*/402.0_km);

  a.foo('A');
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: argument comment missing for literal argument 'chabc' [bugprone-argument-comment]
  // CHECK-FIXES: a.foo(/*chabc=*/'A');

  g(FOO);
  h(1.0f);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'b' [bugprone-argument-comment]
  // CHECK-FIXES: h(/*b=*/1.0f);
  i(__FILE__);

  // FIXME Would like the below to add argument comments.
  g((1));
  // FIXME But we should not add argument comments here.
  g(_Generic(0, int : 0));
}

void f(bool _with_underscores_);
void ignores_underscores() {
  f(false);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument '_with_underscores_' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*_with_underscores_=*/false);

  f(true);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument
  // CHECK-FIXES: f(/*_with_underscores_=*/true);
}
