// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.CommentNullPtrs: true, \
// RUN:     bugprone-argument-comment.IgnoreMacros: false}}" --

#include <stdbool.h>
#include <stddef.h>

#define CHECK(e) do { (void)(e); } while (0)
#define ID(x) x
#define MY_NULL NULL
#define MY_TRUE true
#define CALL_F f(NULL, true)

int f(const char *path, bool force);
int h(const char *path, const char *root);

void test(void) {
  f(NULL, false);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:11: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*path=*/NULL, /*force=*/false);
  f(/*path=*/NULL, /*force=*/false);
  f(/* path= */ NULL, /* force= */ true);

  CHECK(f(NULL, true));
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:17: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES: CHECK(f(/*path=*/NULL, /*force=*/true));
  CHECK(f(/*path=*/NULL, /*force=*/true));

  h(NULL, NULL);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:11: warning: argument comment missing for literal argument 'root' [bugprone-argument-comment]
  // CHECK-FIXES: h(/*path=*/NULL, /*root=*/NULL);
  h(/*path=*/NULL, /*root=*/NULL);
  CHECK(h(/* path= */ NULL, /* root= */ NULL));

  h(NULL, /*rot=*/NULL);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:11: warning: argument name 'rot' in comment does not match parameter name 'root' [bugprone-argument-comment]
  // CHECK-MESSAGES: :[[@LINE-26]]:37: note: 'root' declared here
  // CHECK-FIXES: h(/*path=*/NULL, /*root=*/NULL);

  // The check adds no comment to other macros, to integers, to macros in
  // another macro argument, and to calls in a macro body.
  f(MY_NULL, MY_TRUE);
  f(0, 1);
  f(ID(NULL), ID(true));
  CALL_F;
}
