// RUN: %check_clang_tidy %s modernize-use-nullptr %t -- -- -fno-delayed-template-parsing

// CHECK-FIXES: #include <cstddef>

void foo(decltype(nullptr));
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::nullptr_t instead
// CHECK-FIXES: void foo(std::nullptr_t);
void foo(const decltype(nullptr));
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use std::nullptr_t instead
// CHECK-FIXES: void foo(const std::nullptr_t);
void foo(decltype((nullptr))*);
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use std::nullptr_t instead
// CHECK-FIXES: void foo(std::nullptr_t*);
decltype(nullptr) a;
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: use std::nullptr_t instead
// CHECK-FIXES: std::nullptr_t a;
template<class T=decltype(nullptr)>
struct bar {};
// CHECK-MESSAGES: :[[@LINE-2]]:18: warning: use std::nullptr_t instead
// CHECK-FIXES: template<class T=std::nullptr_t>
