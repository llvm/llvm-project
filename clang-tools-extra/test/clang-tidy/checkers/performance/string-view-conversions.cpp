// RUN: %check_clang_tidy -std=c++17-or-later %s performance-string-view-conversions %t -- \
// RUN:   -- -isystem %clang_tidy_headers

#include <string>

using namespace std::literals::string_literals;
using namespace std::literals::string_view_literals;

// Support for std::move
namespace std {
template <typename>
struct remove_reference;

template <typename _Tp>
struct remove_reference {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &> {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &&> {
  typedef _Tp type;
};

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) {
  return static_cast<typename std::remove_reference<_Tp>::type &&>(__t);
}
} // namespace std

void foo_sv(int p1, std::string_view p2, double p3);
void foo_wsv(int p1, std::wstring_view p2, double p3);
void foo_wsv(std::wstring_view p2);
void foo_str(int p1, const std::string& p2, double p3);
void foo_wstr(int p1, const std::wstring& p2, double p3);

std::string foo_str(int p1);
std::wstring foo_wstr(int, const std::string&);
std::string_view foo_sv(int p1);

void positive(std::string_view sv, std::wstring_view wsv) {
  // string(string_view)
  //
  foo_sv(42, std::string(sv), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, sv, 3.14);

  foo_sv(42, std::string("Hello, world"), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, "Hello, world", 3.14);

  // TODO: support for ""sv literals
  foo_sv(42, "Hello, world"s, 3.14);

  foo_sv(42, std::string{"Hello, world"}, 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, "Hello, world", 3.14);

  const char *ptr = "Hello, world";
  foo_sv(42, std::string(ptr), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, ptr, 3.14);

  char arr[] = "Hello, world";
  foo_sv(42, std::string(arr), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, arr, 3.14);

  foo_sv(42, std::string(foo_sv(42)), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, foo_sv(42), 3.14);

  std::string s = "hello";
  foo_sv(42, std::string(s), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, s, 3.14);

  foo_sv(42, std::string{s}, 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant conversion to 'std::string' (aka 'basic_string<char>') and then back to 'std::string_view' (aka 'basic_string_view<char>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_sv(42, s, 3.14);

  // wstring(wstring_view)
  //
  foo_wsv(42, std::wstring(wsv), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: redundant conversion to 'std::wstring' (aka 'basic_string<wchar_t>') and then back to 'std::wstring_view' (aka 'basic_string_view<wchar_t>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_wsv(42, wsv, 3.14);

  const wchar_t *wptr = L"Hello, world";
  foo_wsv(42, std::wstring(wptr), 3.14);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: redundant conversion to 'std::wstring' (aka 'basic_string<wchar_t>') and then back to 'std::wstring_view' (aka 'basic_string_view<wchar_t>') [performance-string-view-conversions]
  // CHECK-FIXES: foo_wsv(42, wptr, 3.14);
}

void negative(std::string_view sv, std::wstring_view wsv) {
  // No warnings expected: already string_view
  foo_sv(42, sv, 3.14);
  foo_sv(42, "Hello, world", 3.14);
  foo_sv(42, foo_sv(42), 3.14);
  // No warnings expected: complex expression
  foo_sv(42, std::string(sv)  + "bar", 3.14);
  foo_sv(42,
              std::string(    sv  )   +
    ("foo" "bar")    ,
                          3.14);
  foo_sv(42, "foo" + std::string(sv), 3.14);
  foo_sv(42, "foo" + std::string(sv) + "bar", 3.14);
  foo_sv(42, std::string(sv) + std::string(sv), 3.14);
  foo_sv(42, std::string("foo") + std::string("bar"), 3.14);
  foo_sv(42, std::string(5, 'a'), 3.14);
  foo_sv(42, std::string("foo").append("bar"), 3.14);
  foo_sv(42, std::string(sv).substr(0, 5), 3.14);
  foo_sv(42, std::string(sv).c_str(), 3.14);

  // No warnings expected: string parameter, not string-view
  foo_str(42, std::string(sv), 3.14);
  foo_str(42, std::string("Hello, world"), 3.14);
  foo_wstr(42, std::wstring(wsv), 3.14);
  foo_wstr(42, std::wstring(L"Hello, world"), 3.14);

  foo_sv(42, foo_str(42), 3.14);
  foo_sv(42, foo_sv(42), 3.14);

  // Move semantics ignored
  std::string s;
  foo_sv(42, std::move(s), 3.14);

  // Inner calls are ignored
  foo_wsv(foo_wstr(42, "Hello, world"));
  foo_wsv(foo_wstr(42, std::string("Hello, world")));

  // No warnings expected: string parameter of a limited length, not string-view
  foo_sv(142, std::string("Hello, world", 5), 3.14);
}
