// RUN: %check_clang_tidy -std=c++20-or-later %s modernize-use-string-view %t -- -- -isystem %clang_tidy_headers

#include <string>

// ==========================================================
// Positive tests
// ==========================================================

std::string simpleLiteral() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::string_view simpleLiteral() {
  return "simpleLiteral";
}

std::wstring simpleLiteralW() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::wstring_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::wstring_view simpleLiteralW() {
  return L"wide literal";
}

std::u8string simpleLiteral8() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::u8string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::u8string_view simpleLiteral8() {
  return u8"simpleLiteral";
}

std::u16string simpleLiteral16() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::u16string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::u16string_view simpleLiteral16() {
  return u"simpleLiteral";
}

std::u32string simpleLiteral32() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::u32string_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::u32string_view simpleLiteral32() {
  return U"simpleLiteral";
}

// ==========================================================
// Negative tests
// ==========================================================
