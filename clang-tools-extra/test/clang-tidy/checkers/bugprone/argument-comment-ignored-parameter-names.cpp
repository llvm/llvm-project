// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.IgnoredParameterNames: '^[a-z]$;^arg[0-9]*$'}}" --
// RUN: %check_clang_tidy -check-suffix=INVALID %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentBoolLiterals: true, \
// RUN:     bugprone-argument-comment.IgnoredParameterNames: '[;^arg[0-9]*$'}}" --

// CHECK-MESSAGES-INVALID: warning: Invalid regular expression '[' in IgnoredParameterNames [clang-tidy-config]

void f(bool b, bool arg1, bool force);

void test() {
  f(true, false, true);
  // CHECK-MESSAGES: [[@LINE-1]]:18: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-MESSAGES-INVALID: [[@LINE-2]]:5: warning: argument comment missing for literal argument 'b' [bugprone-argument-comment]
  // CHECK-MESSAGES-INVALID: [[@LINE-3]]:18: warning: argument comment missing for literal argument 'force' [bugprone-argument-comment]
  // CHECK-FIXES: f(true, false, /*force=*/true);
  // CHECK-FIXES-INVALID: f(/*b=*/true, false, /*force=*/true);

  f(/*enable=*/true, /*verbose=*/false, /*force=*/true);
  // CHECK-MESSAGES-INVALID: [[@LINE-1]]:5: warning: argument name 'enable' in comment does not match parameter name 'b' [bugprone-argument-comment]
}
