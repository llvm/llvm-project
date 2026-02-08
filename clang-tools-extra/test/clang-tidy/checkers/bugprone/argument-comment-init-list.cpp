// RUN: %check_clang_tidy -check-suffix=OFF -std=c++11,c++14,c++17 %s bugprone-argument-comment %t
// RUN: %check_clang_tidy -check-suffix=ANON -std=c++11,c++14,c++17 %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=TYPED -std=c++11,c++14,c++17 %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffix=BOTH -std=c++11,c++14,c++17 %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --
// RUN: %check_clang_tidy -check-suffixes=BOTH,BOTH-CXX20 -std=c++20-or-later %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-argument-comment.CommentAnonymousInitLists: true, \
// RUN:     bugprone-argument-comment.CommentTypedInitLists: true}}" --

namespace std {
using size_t = decltype(sizeof(0));

template <typename T>
class vector {
public:
  vector();
};

template <typename T>
class initializer_list {
  const T *Begin;
  const T *End;

public:
  initializer_list() : Begin(nullptr), End(nullptr) {}
  const T *begin() const { return Begin; }
  const T *end() const { return End; }
  size_t size() const { return static_cast<size_t>(End - Begin); }
};
} // namespace std

namespace GH171842 {

struct T {
  int value;
};

struct Agg {
  int x;
  int y;
};

void foo(T some_arg, const std::vector<int> &dims);
void foo_init_list(T some_arg, std::initializer_list<int> dims);
void foo_designated(T some_arg, const Agg &dims);

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

  foo(some_arg, {});
  // CHECK-FIXES-OFF: foo(some_arg, {});
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo(some_arg, /*dims=*/{});
  // CHECK-FIXES-TYPED: foo(some_arg, {});
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo(some_arg, /*dims=*/{});

  foo(some_arg, std::vector<int>{});
  // CHECK-FIXES-OFF: foo(some_arg, std::vector<int>{});
  // CHECK-FIXES-ANON: foo(some_arg, std::vector<int>{});
  // CHECK-MESSAGES-TYPED: [[@LINE-3]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-TYPED: foo(some_arg, /*dims=*/std::vector<int>{});
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:17: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo(some_arg, /*dims=*/std::vector<int>{});
}

void test_initializer_list() {
  T some_arg{0};

  foo_init_list(some_arg, {1, 2, 3});
  // CHECK-FIXES-OFF: foo_init_list(some_arg, {1, 2, 3});
  // CHECK-MESSAGES-ANON: [[@LINE-2]]:27: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-ANON: foo_init_list(some_arg, /*dims=*/{1, 2, 3});
  // CHECK-FIXES-TYPED: foo_init_list(some_arg, {1, 2, 3});
  // CHECK-MESSAGES-BOTH: [[@LINE-5]]:27: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH: foo_init_list(some_arg, /*dims=*/{1, 2, 3});
}

#if __cplusplus >= 202002L
void test_designated_init() {
  T some_arg{0};

  foo_designated(some_arg, Agg{.x = 1});
  // CHECK-MESSAGES-BOTH-CXX20: [[@LINE-1]]:28: warning: argument comment missing for literal argument 'dims' [bugprone-argument-comment]
  // CHECK-FIXES-BOTH-CXX20: foo_designated(some_arg, /*dims=*/Agg{.x = 1});
}
#endif

} // namespace GH171842
