// RUN: %check_clang_tidy %s bugprone-string-literal-with-embedded-nul -std=c++11,c++14 %t
// RUN: %check_clang_tidy %s bugprone-string-literal-with-embedded-nul -check-suffix=,CXX17 -std=c++17-or-later %t

namespace std {
using size_t = unsigned long;
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  typedef basic_string<C, T, A> _Type;
  basic_string();
  basic_string(const C *p, const A &a = A());
  basic_string(const C *p, size_t n);
  basic_string(const C *p, size_t n, const A &a);
  template <typename InputIt>
  basic_string(InputIt first, InputIt last);

  _Type& operator+=(const C* s);
  _Type& operator=(const C* s);
};

template <typename C, typename T>
struct basic_string_view {
  basic_string_view(const C *p);
  basic_string_view(const C *p, size_t n);
#if __cplusplus >= 202002L
  template <typename It, typename End>
  basic_string_view(It first, End last);
#endif // __cplusplus >= 202002L
};

typedef basic_string<char, std::char_traits<char>, std::allocator<char>> string;
typedef basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>> wstring;
typedef basic_string_view<char, std::char_traits<char>> string_view;
typedef basic_string_view<wchar_t, std::char_traits<wchar_t>> wstring_view;
} // namespace std

bool operator==(const std::string&, const char*);
bool operator==(const char*, const std::string&);
bool operator==(const std::string_view&, const std::string_view&);


const char Valid[] = "This is valid \x12.";
const char Strange[] = "This is strange \0x12 and must be fixed";
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: suspicious embedded NUL character [bugprone-string-literal-with-embedded-nul]

const char textA[] = "\0x01\0x02\0x03\0x04";
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: suspicious embedded NUL character
const wchar_t textW[] = L"\0x01\0x02\0x03\0x04";
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: suspicious embedded NUL character

const char A[] = "\0";
const char B[] = "\0x";
const char C[] = "\0x1";
const char D[] = "\0x11";
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: suspicious embedded NUL character

const wchar_t E[] = L"\0";
const wchar_t F[] = L"\0x";
const wchar_t G[] = L"\0x1";
const wchar_t H[] = L"\0x11";
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: suspicious embedded NUL character

const char I[] = "\000\000\000\000";
const char J[] = "\0\0\0\0\0\0";
const char K[] = "";

const char L[] = "\0x12" "\0x12" "\0x12" "\0x12";
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: suspicious embedded NUL character

void TestA() {
  std::string str1 = "abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: truncated string literal
  std::string str2 = "\0";
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: truncated string literal
  std::string str3("\0");
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: truncated string literal
  std::string str4{"\x00\x01\x02\x03"};
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: truncated string literal
  std::string str_ok{"abc\0def", 7};
  std::string str_explicit_allocator{"abc\0def", std::allocator<char>()};
  // CHECK-MESSAGES: :[[@LINE-1]]:38: warning: truncated string literal
  std::allocator<char> alloc;
  std::string str_allocator_lvalue{"abc\0def", alloc};
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: truncated string literal
  std::string str_length_allocator_ok{"abc\0def", 7, alloc};
  const char range[] = "abc\0def";
  std::string str_range_ok{range, range + 7};

  std::string str;
  str += "abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: truncated string literal
  str = "abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: truncated string literal

  if (str == "abc\0def") return;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: truncated string literal
  if ("abc\0def" == str) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: truncated string literal
}

void TestW() {
  std::wstring str1 = L"abc\0def";
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: truncated string literal
  std::wstring str2 = L"\0";
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: truncated string literal
  std::wstring str3(L"\0");
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: truncated string literal
  std::wstring str4{L"\x00\x01\x02\x03"};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: truncated string literal
}

#if __cplusplus >= 201703L
void TestView() {
  std::string_view str1 = "abc\0def";
  // CHECK-MESSAGES-CXX17: :[[@LINE-1]]:27: warning: truncated string literal
  std::string_view str2{"abc\0def"};
  // CHECK-MESSAGES-CXX17: :[[@LINE-1]]:25: warning: truncated string literal
  std::string_view str_ok{"abc\0def", 7};

  std::string_view view{"abc"};
  if (view == "abc\0def") return;
  // CHECK-MESSAGES-CXX17: :[[@LINE-1]]:15: warning: truncated string literal
  if ("abc\0def" == view) return;
  // CHECK-MESSAGES-CXX17: :[[@LINE-1]]:7: warning: truncated string literal

  std::wstring_view wstr1 = L"abc\0def";
  // CHECK-MESSAGES-CXX17: :[[@LINE-1]]:29: warning: truncated string literal
  std::wstring_view wstr_ok{L"abc\0def", 7};

#if __cplusplus >= 202002L
  const char range[] = "abc\0def";
  std::string_view str_range_ok{range, range + 7};
#endif // __cplusplus >= 202002L
}
#endif // __cplusplus >= 201703L
