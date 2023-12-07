// RUN: %check_clang_tidy -check-suffix=STDFORMAT -std=c++20 %s readability-redundant-string-cstr %t -- --  -isystem %clang_tidy_headers -DTEST_STDFORMAT
// RUN: %check_clang_tidy -check-suffixes=STDFORMAT,STDPRINT -std=c++2b %s readability-redundant-string-cstr %t -- --  -isystem %clang_tidy_headers -DTEST_STDFORMAT -DTEST_STDPRINT
#include <string>

namespace std {
  template<typename T>
    struct type_identity { using type = T; };
  template<typename T>
    using type_identity_t = typename type_identity<T>::type;

  template <typename CharT, typename... Args>
  struct basic_format_string {
    consteval basic_format_string(const CharT *format) : str(format) {}
    basic_string_view<CharT, std::char_traits<CharT>> str;
  };

  template<typename... Args>
    using format_string = basic_format_string<char, type_identity_t<Args>...>;

  template<typename... Args>
    using wformat_string = basic_format_string<wchar_t, type_identity_t<Args>...>;

#if defined(TEST_STDFORMAT)
  template<typename ...Args>
  std::string format(format_string<Args...>, Args &&...);
  template<typename ...Args>
  std::string format(wformat_string<Args...>, Args &&...);
#endif // TEST_STDFORMAT

#if defined(TEST_STDPRINT)
  template<typename ...Args>
  void print(format_string<Args...>, Args &&...);
  template<typename ...Args>
  void print(wformat_string<Args...>, Args &&...);
#endif // TEST_STDPRINT
}

namespace notstd {
#if defined(TEST_STDFORMAT)
  template<typename ...Args>
  std::string format(const char *, Args &&...);
  template<typename ...Args>
  std::string format(const wchar_t *, Args &&...);
#endif // TEST_STDFORMAT
#if defined(TEST_STDPRINT)
  template<typename ...Args>
  void print(const char *, Args &&...);
  template<typename ...Args>
  void print(const wchar_t *, Args &&...);
#endif // TEST_STDPRINT
}

std::string return_temporary();
std::wstring return_wtemporary();

#if defined(TEST_STDFORMAT)
void std_format(const std::string &s1, const std::string &s2, const std::string &s3) {
  auto r1 = std::format("One:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-1]]:37: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDFORMAT: {{^  }}auto r1 = std::format("One:{}\n", s1);

  auto r2 = std::format("One:{} Two:{} Three:{} Four:{}\n", s1.c_str(), s2, s3.c_str(), return_temporary().c_str());
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-1]]:61: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-2]]:77: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-3]]:89: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDFORMAT: {{^  }}auto r2 = std::format("One:{} Two:{} Three:{} Four:{}\n", s1, s2, s3, return_temporary());

  using namespace std;
  auto r3 = format("Four:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-1]]:33: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDFORMAT: {{^  }}auto r3 = format("Four:{}\n", s1);
}

void std_format_wide(const std::wstring &s1, const std::wstring &s2, const std::wstring &s3) {
  auto r1 = std::format(L"One:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-1]]:38: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDFORMAT: {{^  }}auto r1 = std::format(L"One:{}\n", s1);

  auto r2 = std::format(L"One:{} Two:{} Three:{} Four:{}\n", s1.c_str(), s2, s3.c_str(), return_wtemporary().c_str());
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-1]]:62: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-2]]:78: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-3]]:90: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDFORMAT: {{^  }}auto r2 = std::format(L"One:{} Two:{} Three:{} Four:{}\n", s1, s2, s3, return_wtemporary());

  using namespace std;
  auto r3 = format(L"Four:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDFORMAT: :[[@LINE-1]]:34: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDFORMAT: {{^  }}auto r3 = format(L"Four:{}\n", s1);
}

// There's are c_str() calls here, so it shouldn't be touched.
std::string std_format_no_cstr(const std::string &s1, const std::string &s2) {
  return std::format("One: {}, Two: {}\n", s1, s2);
}

// There's are c_str() calls here, so it shouldn't be touched.
std::string std_format_no_cstr_wide(const std::string &s1, const std::string &s2) {
  return std::format(L"One: {}, Two: {}\n", s1, s2);
}

// This is not std::format, so it shouldn't be fixed.
std::string not_std_format(const std::string &s1) {
  return notstd::format("One: {}\n", s1.c_str());

  using namespace notstd;
  format("One: {}\n", s1.c_str());
}

// This is not std::format, so it shouldn't be fixed.
std::string not_std_format_wide(const std::string &s1) {
  return notstd::format(L"One: {}\n", s1.c_str());

  using namespace notstd;
  format(L"One: {}\n", s1.c_str());
}
#endif // TEST_STDFORMAT

#if defined(TEST_STDPRINT)
void std_print(const std::string &s1, const std::string &s2, const std::string &s3) {
  std::print("One:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-1]]:26: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDPRINT: {{^  }}std::print("One:{}\n", s1);

  std::print("One:{} Two:{} Three:{} Four:{}\n", s1.c_str(), s2, s3.c_str(), return_temporary().c_str());
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-1]]:50: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-2]]:66: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-3]]:78: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDPRINT: {{^  }}std::print("One:{} Two:{} Three:{} Four:{}\n", s1, s2, s3, return_temporary());

  using namespace std;
  print("Four:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-1]]:22: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDPRINT: {{^  }}print("Four:{}\n", s1);
}

void std_print_wide(const std::wstring &s1, const std::wstring &s2, const std::wstring &s3) {
  std::print(L"One:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-1]]:27: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDPRINT: {{^  }}std::print(L"One:{}\n", s1);

  std::print(L"One:{} Two:{} Three:{} Four:{}\n", s1.c_str(), s2, s3.c_str(), return_wtemporary().c_str());
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-1]]:51: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-2]]:67: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-3]]:79: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDPRINT: {{^  }}std::print(L"One:{} Two:{} Three:{} Four:{}\n", s1, s2, s3, return_wtemporary());

  using namespace std;
  print(L"Four:{}\n", s1.c_str());
  // CHECK-MESSAGES-STDPRINT: :[[@LINE-1]]:23: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES-STDPRINT: {{^  }}print(L"Four:{}\n", s1);
}

// There's no c_str() call here, so it shouldn't be touched.
void std_print_no_cstr(const std::string &s1, const std::string &s2) {
  std::print("One: {}, Two: {}\n", s1, s2);
}

// There's no c_str() call here, so it shouldn't be touched.
void std_print_no_cstr_wide(const std::wstring &s1, const std::wstring &s2) {
  std::print(L"One: {}, Two: {}\n", s1, s2);
}

// This isn't std::print, so it shouldn't be fixed.
void not_std_print(const std::string &s1) {
  notstd::print("One: {}\n", s1.c_str());

  using namespace notstd;
  print("One: {}\n", s1.c_str());
}

// This isn't std::print, so it shouldn't be fixed.
void not_std_print_wide(const std::string &s1) {
  notstd::print(L"One: {}\n", s1.c_str());

  using namespace notstd;
  print(L"One: {}\n", s1.c_str());
}
#endif // TEST_STDPRINT

#if defined(TEST_STDFORMAT)
// We can't declare these earlier since they make the "using namespace std"
// tests ambiguous.
template<typename ...Args>
std::string format(const char *, Args &&...);
template<typename ...Args>
std::string format(const wchar_t *, Args &&...);

// This is not std::format, so it shouldn't be fixed.
std::string not_std_format2(const std::wstring &s1) {
  return format("One: {}\n", s1.c_str());
}

// This is not std::format, so it shouldn't be fixed.
std::string not_std_format2_wide(const std::wstring &s1) {
  return format(L"One: {}\n", s1.c_str());
}
#endif // TEST_STDFORMAT

#if defined(TEST_STDPRINT)
template<typename ...Args>
void print(const char *, Args &&...);
template<typename ...Args>
void print(const wchar_t *, Args &&...);

// This isn't std::print, so it shouldn't be fixed.
void not_std_print2(const std::string &s1) {
  print("One: {}\n", s1.c_str());
}

// This isn't std::print, so it shouldn't be fixed.
void not_std_print2_wide(const std::string &s1) {
  print(L"One: {}\n", s1.c_str());
}
#endif // TEST_STDPRINT
