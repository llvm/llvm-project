// RUN: %check_clang_tidy -check-suffixes=,MACROS %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.CommentIntegerLiterals: true, \
// RUN:     bugprone-argument-comment.IgnoreMacros: false}}" --
// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.CommentIntegerLiterals: true}}" --

struct A {
  void foo(bool abc);
};

void g(int a);
void j(int a, int b, int c);
void k(int a, int b);
void real_g(int a, bool b);

#define CHECK(e) do { (void)(e); } while (0)
#define CHECK_TWICE(e) do { (void)(e); (void)(e); } while (0)
#define ID(x) x
#define X(x) (x)
#define VOID(e) ((void)(e))
#define CALL_G(x) g(x)
#define CALL_G_1 g(1)
#define CALL(fn, a) fn(a)
#define CALL2(fn, a) fn(a, a)
#define my_g real_g

void test() {
  A a;

  my_g(0, true);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: argument comment missing for literal argument 'a' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:11: warning: argument comment missing for literal argument 'b' [bugprone-argument-comment]
  // CHECK-FIXES: my_g(/*a=*/0, /*b=*/true);

  ID(real_g)(0, true);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: argument comment missing for literal argument 'a' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:17: warning: argument comment missing for literal argument 'b' [bugprone-argument-comment]
  // CHECK-FIXES: ID(real_g)(/*a=*/0, /*b=*/true);

  CHECK(a.foo(true));
  // CHECK-MESSAGES-MACROS: [[@LINE-1]]:15: warning: argument comment missing for literal argument 'abc' [bugprone-argument-comment]
  // CHECK-FIXES-MACROS: CHECK(a.foo(/*abc=*/true));
  CHECK(a.foo(/*abc=*/true));

  CHECK(j(1, X(1), -1));
  // CHECK-MESSAGES-MACROS: [[@LINE-1]]:11: warning: argument comment missing for literal argument 'a' [bugprone-argument-comment]
  // CHECK-MESSAGES-MACROS: [[@LINE-2]]:20: warning: argument comment missing for literal argument 'c' [bugprone-argument-comment]
  // CHECK-FIXES-MACROS: CHECK(j(/*a=*/1, X(1), /*c=*/-1));

  CHECK_TWICE(g(1));
  // CHECK-MESSAGES-MACROS: [[@LINE-1]]:17: warning: argument comment missing for literal argument 'a' [bugprone-argument-comment]
  // CHECK-FIXES-MACROS: CHECK_TWICE(g(/*a=*/1));

  CHECK(VOID(g(1)));
  // CHECK-MESSAGES-MACROS: [[@LINE-1]]:16: warning: argument comment missing for literal argument 'a' [bugprone-argument-comment]
  // CHECK-FIXES-MACROS: CHECK(VOID(g(/*a=*/1)));

  // The check adds no comment when the literal and the closing parenthesis of
  // the call are in different macro arguments, or when the call is in a macro
  // body.
  g(ID(1));
  CHECK(g(ID(1)));
  ID(g)(ID(1));
  CALL_G(1);
  CALL_G_1;
  CALL(g, 1);
  CALL2(k, 1);
}
