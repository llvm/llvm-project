// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-constraints %t -- -- -fno-delayed-template-parsing

#include <utility>

// Separate test file for the case where the first '>>' token part of
// an enable_if expression correctly handles the synthesized token.

template <typename T, typename = std::enable_if_t<T::some_value>>
void first_greatergreater_is_enable_if() {
}
// CHECK-MESSAGES: :[[@LINE-3]]:23: warning: use C++20 requires constraints instead of enable_if [modernize-use-constraints]
// CHECK-FIXES: void first_greatergreater_is_enable_if() requires T::some_value {
