// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentIntegerLiterals: ZeroOnly}}" --
// RUN: %check_clang_tidy -check-suffix=ALL %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentIntegerLiterals: All}}" --
// RUN: %check_clang_tidy -check-suffix=INVALID %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentIntegerLiterals: Sometimes}}" --

// CHECK-MESSAGES-INVALID: warning: invalid configuration value 'Sometimes' for option 'bugprone-argument-comment.CommentIntegerLiterals' [clang-tidy-config]

void f(int flags, unsigned long size, double ratio, bool force);

#define ZERO 0

void test() {
  f(0, 0UL, 0.0, false);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:8: warning: argument comment missing for literal argument 'size' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-3]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-4]]:8: warning: argument comment missing for literal argument 'size' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*flags=*/0, /*size=*/0UL, 0.0, false);
  // CHECK-FIXES-ALL: f(/*flags=*/0, /*size=*/0UL, 0.0, false);

  f(0x0, 4096, 1.0, true);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-2]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-3]]:10: warning: argument comment missing for literal argument 'size' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*flags=*/0x0, 4096, 1.0, true);
  // CHECK-FIXES-ALL: f(/*flags=*/0x0, /*size=*/4096, 1.0, true);

  f(1, 4096, 0.0, true);
  // CHECK-MESSAGES-ALL: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-2]]:8: warning: argument comment missing for literal argument 'size' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: f(/*flags=*/1, /*size=*/4096, 0.0, true);

  f(~0, -0, 0.0, !0);
  // CHECK-MESSAGES-ALL: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-2]]:9: warning: argument comment missing for literal argument 'size' [bugprone-argument-comment]
  // CHECK-MESSAGES-ALL: [[@LINE-3]]:18: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: f(/*flags=*/~0, /*size=*/-0, 0.0, /*force=*/!0);

  f(/*flags=*/0, /*size=*/0, 0.0, false);
  f(ZERO, ZERO, 0.0, false);
}
