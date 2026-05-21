// RUN: %check_clang_tidy -check-suffix=OFF -std=c++20-or-later %s bugprone-argument-comment %t
// RUN: %check_clang_tidy -check-suffix=ANON -std=c++20-or-later %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=TYPED -std=c++20-or-later %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=TEMP -std=c++20-or-later %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentParenthesizedTemporaries: true}}" --
// RUN: %check_clang_tidy -check-suffix=BOTH-INIT -std=c++20-or-later %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=ALL -std=c++20-or-later %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true, \
// RUN:     bugprone-argument-comment.CommentParenthesizedTemporaries: true}}" --

struct T {
  int value;
};

struct Agg {
  int x;
  int y;
};

void foo_designated(T some_arg, const Agg &dims);

void test_designated_init() {
  T some_arg{0};

  foo_designated(some_arg, /*dim=*/Agg{.x = 1});
  // CHECK-MESSAGES-OFF: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-OFF: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-ANON: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ANON: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-TYPED: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TYPED: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-TEMP: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TEMP: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-BOTH-INIT: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-BOTH-INIT: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-ALL: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ALL: foo_designated(some_arg, /*dims=*/Agg{.x = 1});

  foo_designated(some_arg, Agg{.x = 1});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:28: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:28: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:28: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_designated(some_arg, /*dims=*/Agg{.x = 1});

  foo_designated(some_arg, Agg(1, 2));
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-3]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP: [[@LINE-4]]:28: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TEMP: foo_designated(some_arg, /*dims=*/Agg(1, 2));
  // CHECK-MESSAGES-BOTH-INIT-NOT: :[[@LINE-6]]:28: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ALL: [[@LINE-7]]:28: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_designated(some_arg, /*dims=*/Agg(1, 2));
}
