// RUN: %check_clang_tidy -std=c++20-or-later %s performance-string-view-conversions %t -- \
// RUN:   -- -isystem %clang_tidy_headers

#include <string>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

void foo_u8sv(int p1, std::u8string_view p2, double p3);
void foo_u16sv(int p1, std::u16string_view p2, double p3);
void foo_u32sv(int p1, std::u32string_view p2, double p3);

void positive(std::string_view sv, std::wstring_view wsv) {
  // [u8|u16|32]string([u8|u16|32]string_view)
  //
  foo_u8sv(42, std::u8string(u8"Hello, world"), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant conversion to 'std::u8string' (aka 'basic_string<char8_t>') and then back to 'std::u8string_view' (aka 'basic_string_view<char8_t>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_u8sv(42, u8"Hello, world", 3.14);

  foo_u16sv(42, std::u16string(u"Hello, world"), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant conversion to 'std::u16string' (aka 'basic_string<char16_t>') and then back to 'std::u16string_view' (aka 'basic_string_view<char16_t>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_u16sv(42, u"Hello, world", 3.14);

  foo_u32sv(42, std::u32string(U"Hello, world"), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant conversion to 'std::u32string' (aka 'basic_string<char32_t>') and then back to 'std::u32string_view' (aka 'basic_string_view<char32_t>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_u32sv(42, U"Hello, world", 3.14);
}
