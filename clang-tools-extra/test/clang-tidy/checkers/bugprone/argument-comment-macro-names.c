// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.CommentNullPtrs: true, \
// RUN:     bugprone-argument-comment.NullPtrMacros: 'NULL;MY_NULL', \
// RUN:     bugprone-argument-comment.BoolLiteralMacros: 'TRUE;FALSE'}}" --

#include <stddef.h>

#define MY_NULL ((void *)0)
#define TRUE (1 == 1)
#define FALSE 0
#define OTHER 0

int f(const char *path, int force);

void test(void) {
  f(MY_NULL, TRUE);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:14: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*path=*/MY_NULL, /*force=*/TRUE);

  f(NULL, FALSE);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:11: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*path=*/NULL, /*force=*/FALSE);

  f(NULL, OTHER);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*path=*/NULL, OTHER);
}
