// RUN: %check_clang_tidy -std=gnu++98 %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.CommentNullPtrs: true}}" --

// In GNU C++98, <stdbool.h> defines `true` and `false` as macros, and
// <stddef.h> defines `NULL` as `__null`.
#include <stdbool.h>
#include <stddef.h>

void f(const char *path, bool force);

void test() {
  f(NULL, true);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'path' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:11: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*path=*/NULL, /*force=*/true);
}
