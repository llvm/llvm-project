// RUN: %check_clang_tidy \
// RUN: -std=c++17-or-later %s modernize-use-string-view %t -- \
// RUN: --config="{CheckOptions: {modernize-use-string-view.ReplacementStringViewClass: 'string=StringRef;u8string=U8StringRef'}}" \
// RUN: -- -isystem %clang_tidy_headers

#include <string>

// ==========================================================
// Positive tests
// ==========================================================

std::string Literal() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'StringRef' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: StringRef Literal() {
  return "literal";
}

std::wstring WLiteral() {
// CHECK-MESSAGES:[[@LINE-1]]:1: warning: consider using 'std::wstring_view' to avoid unnecessary copying and allocations [modernize-use-string-view]
// CHECK-FIXES: std::wstring_view WLiteral() {
  return L"literal";
}
