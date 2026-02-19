// RUN: %check_clang_tidy \
// RUN: -std=c++17-or-later %s modernize-use-string-view %t -- \
// RUN: --config="{CheckOptions: {modernize-use-string-view.IgnoredFunctions: 'GoodButIgnored;GoodTooButAlsoIgnored'}}" \
// RUN: -- -isystem %clang_tidy_headers

#include <string>

// ==========================================================
// Positive tests
// ==========================================================

std::string toString() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view toString() {
  return "not ignored by custom options";
}

std::string ToString() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view ToString() {
  return "not ignored by custom options";
}

std::string to_string() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view to_string() {
  return "not ignored by custom options";
}

// ==========================================================
// Negative tests
// ==========================================================

std::string GoodButIgnored() {
  return "ignored by explicit options";
}

std::string GoodTooButAlsoIgnored() {
  return "also ignored by explicit options";
}
