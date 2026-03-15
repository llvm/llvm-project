// RUN: %check_clang_tidy -check-suffix=OFF %s bugprone-argument-comment %t
// RUN: %check_clang_tidy -check-suffix=ANON %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=TYPED %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=BOTH %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --

#include <initializer_list>
#include <vector>

namespace GH171842 {

struct T {
  int value;
};

void foo(T some_arg, const std::vector<int> &dims);
void foo_init_list(T some_arg, std::initializer_list<int> dims);
template <typename ElemTy>
void foo_template(T some_arg, const std::vector<ElemTy> &dims);

void test_braced_init_list() {
  T some_arg{0};

  // Mismatched explicit argument comments are validated independently of the
  // init-list literal comment options.
  foo(some_arg, /*dim=*/{});
  // CHECK-MESSAGES-OFF: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-OFF: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-ANON: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TYPED: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-BOTH: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-BOTH: foo(some_arg, /*dims=*/{});

  foo(some_arg, /*dim=*/std::vector<int>{});
  // CHECK-MESSAGES-OFF: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-OFF: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-ANON: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-TYPED: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-BOTH: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-BOTH: foo(some_arg, /*dims=*/std::vector<int>{});

  foo(some_arg, {});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:17: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:17: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo(some_arg, /*dims=*/{});

  foo(some_arg, std::vector<int>{});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:17: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:17: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo(some_arg, /*dims=*/std::vector<int>{});
}

void test_initializer_list() {
  T some_arg{0};

  foo_init_list(some_arg, {1, 2, 3});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:27: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:27: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo_init_list(some_arg, /*dims=*/{1, 2, 3});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:27: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:27: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo_init_list(some_arg, /*dims=*/{1, 2, 3});
}

template <typename ElemTy>
void test_template_dependent_init_list() {
  T some_arg{0};

  foo_template<ElemTy>(some_arg, {});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:34: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:34: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo_template<ElemTy>(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:34: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:34: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo_template<ElemTy>(some_arg, /*dims=*/{});

  foo_template<ElemTy>(some_arg, std::vector<ElemTy>{});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:34: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:34: warning: argument comment missing for literal argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:34: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo_template<ElemTy>(some_arg, /*dims=*/std::vector<ElemTy>{});
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:34: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo_template<ElemTy>(some_arg, /*dims=*/std::vector<ElemTy>{});
}

template void test_template_dependent_init_list<int>();

} // namespace GH171842
