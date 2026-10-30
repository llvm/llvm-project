// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentZeroIntegerLiterals: true}}" --

void f(int flags, unsigned long size, double ratio, bool force);

#define ZERO 0

void test() {
  f(0, 0UL, 0.0, false);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-MESSAGES: [[@LINE-2]]:8: warning: argument comment missing for literal argument 'size' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*flags=*/0, /*size=*/0UL, 0.0, false);

  f(0x0, 4096, 1.0, true);
  // CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-FIXES: f(/*flags=*/0x0, 4096, 1.0, true);

  f(/*flags=*/0, /*size=*/0, 0.0, false);
  f(1, 4096, 0.0, true);
  f(ZERO, ZERO, 0.0, false);
  f(~0, -0, 0.0, !0);
}
