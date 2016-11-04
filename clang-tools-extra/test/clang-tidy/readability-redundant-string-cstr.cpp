// RUN: %check_clang_tidy %s readability-redundant-string-cstr %t -- -- -std=c++11

typedef unsigned __INT16_TYPE__ char16;
typedef unsigned __INT32_TYPE__ char32;
typedef __SIZE_TYPE__ size;

namespace std {
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  typedef basic_string<C, T, A> _Type;
  basic_string();
  basic_string(const C *p, const A &a = A());

  const C *c_str() const;
  const C *data() const;

  _Type& append(const C *s);
  _Type& append(const C *s, size n);
  _Type& assign(const C *s);
  _Type& assign(const C *s, size n);

  int compare(const _Type&) const;
  int compare(const C* s) const;
  int compare(size pos, size len, const _Type&) const;
  int compare(size pos, size len, const C* s) const;

  size find(const _Type& str, size pos = 0) const;
  size find(const C* s, size pos = 0) const;
  size find(const C* s, size pos, size n) const;

  _Type& insert(size pos, const _Type& str);
  _Type& insert(size pos, const C* s);
  _Type& insert(size pos, const C* s, size n);

  _Type& operator+=(const _Type& str);
  _Type& operator+=(const C* s);
  _Type& operator=(const _Type& str);
  _Type& operator=(const C* s);
};

typedef basic_string<char, std::char_traits<char>, std::allocator<char>> string;
typedef basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t>> wstring;
typedef basic_string<char16, std::char_traits<char16>, std::allocator<char16>> u16string;
typedef basic_string<char32, std::char_traits<char32>, std::allocator<char32>> u32string;
}

std::string operator+(const std::string&, const std::string&);
std::string operator+(const std::string&, const char*);
std::string operator+(const char*, const std::string&);

bool operator==(const std::string&, const std::string&);
bool operator==(const std::string&, const char*);
bool operator==(const char*, const std::string&);

namespace llvm {
struct StringRef {
  StringRef(const char *p);
  StringRef(const std::string &);
};
}

// Tests for std::string.

void f1(const std::string &s) {
  f1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f1(s);{{$}}
  f1(s.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'data' [readability-redundant-string-cstr]
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
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f1(*ptr);{{$}}
}
void f5(const std::string &s) {
  std::string tmp;
  tmp.append(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp.append(s);{{$}}
  tmp.assign(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp.assign(s);{{$}}

  if (tmp.compare(s.c_str()) == 0) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}if (tmp.compare(s) == 0) return;{{$}}

  if (tmp.compare(1, 2, s.c_str()) == 0) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}if (tmp.compare(1, 2, s) == 0) return;{{$}}

  if (tmp.find(s.c_str()) == 0) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}if (tmp.find(s) == 0) return;{{$}}

  if (tmp.find(s.c_str(), 2) == 0) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}if (tmp.find(s, 2) == 0) return;{{$}}

  if (tmp.find(s.c_str(), 2) == 0) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}if (tmp.find(s, 2) == 0) return;{{$}}

  tmp.insert(1, s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp.insert(1, s);{{$}}

  tmp = s.c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp = s;{{$}}

  tmp += s.c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp += s;{{$}}

  if (tmp == s.c_str()) return;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}if (tmp == s) return;{{$}}

  tmp = s + s.c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp = s + s;{{$}}

  tmp = s.c_str() + s;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant call {{.*}}
  // CHECK-FIXES: {{^  }}tmp = s + s;{{$}}
}
void f6(const std::string &s) {
  std::string tmp;
  tmp.append(s.c_str(), 2);
  tmp.assign(s.c_str(), 2);

  if (tmp.compare(s) == 0) return;
  if (tmp.compare(1, 2, s) == 0) return;

  tmp = s;
  tmp += s;

  if (tmp == s)
    return;

  tmp = s + s;

  if (tmp.find(s.c_str(), 2, 4) == 0) return;

  tmp.insert(1, s);
  tmp.insert(1, s.c_str(), 2);
}

// Tests for std::wstring.

void g1(const std::wstring &s) {
  g1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}g1(s);{{$}}
}

// Tests for std::u16string.

void h1(const std::u16string &s) {
  h1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}h1(s);{{$}}
}

// Tests for std::u32string.

void k1(const std::u32string &s) {
  k1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
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
