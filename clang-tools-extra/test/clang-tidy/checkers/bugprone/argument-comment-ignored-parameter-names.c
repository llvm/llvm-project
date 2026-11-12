// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentIntegerLiterals: true, \
// RUN:     bugprone-argument-comment.CommentStringLiterals: true, \
// RUN:     bugprone-argument-comment.IgnoredParameterNames: '^[a-z]$;^arg[0-9]*$;^__'}}" --

int open_at(int d, const char *p, int flags, int __mode);
int call(int arg1, int arg2, int timeout);

void test(void) {
  open_at(3, "f", 0, 0644);
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: argument comment missing for literal argument 'flags' [bugprone-argument-comment]
  // CHECK-FIXES: open_at(3, "f", /*flags=*/0, 0644);

  call(1, 2, 30);
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: argument comment missing for literal argument 'timeout' [bugprone-argument-comment]
  // CHECK-FIXES: call(1, 2, /*timeout=*/30);

  open_at(/*dirfd=*/3, /*path=*/"f", /*flags=*/0, /*mode=*/0644);
  call(/*first=*/1, /*second=*/2, /*timeout=*/30);
}
