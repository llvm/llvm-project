// RUN: %check_clang_tidy -check-suffix=OFF %s bugprone-argument-comment %t
// RUN: %check_clang_tidy -check-suffix=ANON %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=TYPED %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=TEMP %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentParenthesizedTemporaries: true}}" --
// RUN: %check_clang_tidy -check-suffix=BOTH-INIT %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=ALL %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true, \
// RUN:     bugprone-argument-comment.CommentParenthesizedTemporaries: true}}" --

#include <initializer_list>
#include <vector>

struct T {
  int value;
};

struct Dims {
  Dims();
  Dims(int, int, int);
};

void foo(T some_arg, const std::vector<int> &dims);
void foo_dims(T some_arg, const Dims &dims);
void foo_init_list(T some_arg, std::initializer_list<int> dims);
void foo_nested_init_list(T some_arg,
                          std::initializer_list<std::initializer_list<int>> dims);
void foo_int(T some_arg, int value);
template <typename ElemTy>
void foo_template(T some_arg, const std::vector<ElemTy> &dims);
template <typename DimsTy>
void foo_template_typed(T some_arg, const DimsTy &dims);

void test_braced_init_list() {
  T some_arg{0};

  // Mismatched explicit argument comments are validated independently of the
  // missing-comment options.
  foo(some_arg, /*dim=*/{});
  // CHECK-MESSAGES-OFF: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-OFF: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-ANON: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TYPED: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TEMP: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TEMP: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-BOTH-INIT: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-BOTH-INIT: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-ALL: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ALL: foo(some_arg, /*dims=*/{});

  foo(some_arg, /*dim=*/std::vector<int>{});
  // CHECK-MESSAGES-OFF: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-OFF: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-ANON: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-TYPED: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-TEMP: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TEMP: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-BOTH-INIT: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-BOTH-INIT: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-ALL: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ALL: foo(some_arg, /*dims=*/std::vector<int>{});

  foo(some_arg, {});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:17: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:17: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:17: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:17: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:17: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:17: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo(some_arg, /*dims=*/{});

  foo(some_arg, std::vector<int>{});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:17: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:17: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:17: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:17: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:17: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:17: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo(some_arg, /*dims=*/std::vector<int>{});
}

void test_initializer_list() {
  T some_arg{0};

  foo_init_list(some_arg, {1, 2, 3});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:27: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:27: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo_init_list(some_arg, /*dims=*/{1, 2, 3});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:27: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:27: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:27: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_init_list(some_arg, /*dims=*/{1, 2, 3});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:27: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_init_list(some_arg, /*dims=*/{1, 2, 3});

  foo_init_list(some_arg, std::initializer_list<int>{1, 2, 3});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:27: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:27: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:27: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo_init_list(some_arg, /*dims=*/std::initializer_list<int>{1, 2, 3});
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:27: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:27: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_init_list(some_arg, /*dims=*/std::initializer_list<int>{1, 2, 3});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:27: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_init_list(some_arg, /*dims=*/std::initializer_list<int>{1, 2, 3});
}

void test_nested_initializer_list() {
  T some_arg{0};

  foo_nested_init_list(some_arg, {{1, 2}, {3, 4}});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo_nested_init_list(some_arg, /*dims=*/{{.*}});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_nested_init_list(some_arg, /*dims=*/{{.*}});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_nested_init_list(some_arg, /*dims=*/{{.*}});
}

void test_parenthesized_temporary() {
  T some_arg{0};

  foo_dims(some_arg, /*dim=*/Dims());
  // CHECK-MESSAGES-OFF: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-OFF: foo_dims(some_arg, /*dims=*/Dims());
  // CHECK-MESSAGES-ANON: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ANON: foo_dims(some_arg, /*dims=*/Dims());
  // CHECK-MESSAGES-TYPED: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TYPED: foo_dims(some_arg, /*dims=*/Dims());
  // CHECK-MESSAGES-TEMP: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-TEMP: foo_dims(some_arg, /*dims=*/Dims());
  // CHECK-MESSAGES-BOTH-INIT: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-BOTH-INIT: foo_dims(some_arg, /*dims=*/Dims());
  // CHECK-MESSAGES-ALL: warning: argument name 'dim' in comment does not match parameter name 'dims'
  // CHECK-FIXES-ALL: foo_dims(some_arg, /*dims=*/Dims());

  foo_dims(some_arg, Dims{});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo_dims(some_arg, /*dims=*/Dims{});
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_dims(some_arg, /*dims=*/Dims{});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_dims(some_arg, /*dims=*/Dims{});

  foo_dims(some_arg, Dims());
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-3]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP: [[@LINE-4]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TEMP: foo_dims(some_arg, /*dims=*/Dims());
  // CHECK-MESSAGES-BOTH-INIT-NOT: :[[@LINE-6]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ALL: [[@LINE-7]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_dims(some_arg, /*dims=*/Dims());

  foo_dims(some_arg, Dims(1, 2, 3));
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-3]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP: [[@LINE-4]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TEMP: foo_dims(some_arg, /*dims=*/Dims(1, 2, 3));
  // CHECK-MESSAGES-BOTH-INIT-NOT: :[[@LINE-6]]:22: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ALL: [[@LINE-7]]:22: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_dims(some_arg, /*dims=*/Dims(1, 2, 3));

  foo_int(some_arg, int(1));
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:21: warning: argument comment missing for argument 'value'
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-2]]:21: warning: argument comment missing for argument 'value'
  // CHECK-MESSAGES-ALL-NOT: :[[@LINE-3]]:21: warning: argument comment missing for argument 'value'
}

template <typename ElemTy>
void test_template_dependent_init_list() {
  T some_arg{0};

  foo_template<ElemTy>(some_arg, {});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo_template<ElemTy>(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-TYPED-NOT: :[[@LINE-4]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_template<ElemTy>(some_arg, /*dims=*/{});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_template<ElemTy>(some_arg, /*dims=*/{});

  foo_template<ElemTy>(some_arg, std::vector<ElemTy>{});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo_template<ElemTy>(some_arg, /*dims=*/std::vector<ElemTy>{});
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:34: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_template<ElemTy>(some_arg, /*dims=*/std::vector<ElemTy>{});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:34: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_template<ElemTy>(some_arg, /*dims=*/std::vector<ElemTy>{});
}

template <typename DimsTy>
void test_template_dependent_typed_init_list() {
  T some_arg{0};

  foo_template_typed<DimsTy>(some_arg, DimsTy{1, 2, 3});
  // CHECK-MESSAGES-OFF-NOT: :[[@LINE-1]]:40: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-ANON-NOT: :[[@LINE-2]]:40: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:40: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo_template_typed<DimsTy>(some_arg, /*dims=*/DimsTy{1, 2, 3});
  // CHECK-MESSAGES-TEMP-NOT: :[[@LINE-5]]:40: warning: argument comment missing for argument 'dims'
  // CHECK-MESSAGES-BOTH-INIT: [[@LINE-6]]:40: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-INIT: foo_template_typed<DimsTy>(some_arg, /*dims=*/DimsTy{1, 2, 3});
  // CHECK-MESSAGES-ALL: [[@LINE-8]]:40: warning: argument comment missing for argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ALL: foo_template_typed<DimsTy>(some_arg, /*dims=*/DimsTy{1, 2, 3});
}

template void test_template_dependent_init_list<int>();
template void test_template_dependent_typed_init_list<std::vector<int>>();
