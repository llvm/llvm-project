// RUN: %check_clang_tidy %s readability-redundant-string-cstr %t -- -- -target x86_64-unknown -std=c++11

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  basic_string();
  basic_string(const C *p, const A &a = A());
  const C *c_str() const;
};
typedef basic_string<char, std::char_traits<char>, std::allocator<char>> string;
typedef basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>> wstring;
typedef basic_string<char16_t, std::char_traits<char16_t>, std::allocator<char16_t>> u16string;
typedef basic_string<char32_t, std::char_traits<char32_t>, std::allocator<char32_t>> u32string;
}

namespace llvm {
struct StringRef {
  StringRef(const char *p);
  StringRef(const std::string &);
};
}

// Tests for std::string.

void f1(const std::string &s) {
  f1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to `c_str()` [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f1(s);{{$}}
}
void f2(const llvm::StringRef r) {
  std::string s;
  f2(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}std::string s;{{$}}
  // CHECK-FIXES-NEXT: {{^  }}f2(s);{{$}}
}
void f3(const llvm::StringRef &r) {
  std::string s;
  f3(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}std::string s;{{$}}
  // CHECK-FIXES-NEXT: {{^  }}f3(s);{{$}}
}
void f4(const std::string &s) {
  const std::string* ptr = &s;
  f1(ptr->c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to `c_str()` [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f1(*ptr);{{$}}
}

// Tests for std::wstring.

void g1(const std::wstring &s) {
  g1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to `c_str()` [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}g1(s);{{$}}
}

// Tests for std::u16string.

void h1(const std::u16string &s) {
  h1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to `c_str()` [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}h1(s);{{$}}
}

// Tests for std::u32string.

void k1(const std::u32string &s) {
  k1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to `c_str()` [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}k1(s);{{$}}
}

// Tests on similar classes that aren't good candidates for this checker.

struct NotAString {
  NotAString();
  NotAString(const NotAString&);
  const char *c_str() const;
};

void dummy(const char*) {}

void invalid(const NotAString &s) {
  dummy(s.c_str());
}
