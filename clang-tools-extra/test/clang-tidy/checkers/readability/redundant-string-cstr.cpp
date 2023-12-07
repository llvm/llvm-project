// RUN: %check_clang_tidy %s readability-redundant-string-cstr %t -- -- -isystem %clang_tidy_headers
#include <string>

template <typename T>
struct iterator {
  T *operator->();
  T &operator*();
};

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
void f7(std::string_view sv) {
  std::string s;
  f7(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f7(s);{{$}}
  f7(s.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'data' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}f7(s);{{$}}
}

// Tests for std::wstring.

void g1(const std::wstring &s) {
  g1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}g1(s);{{$}}
}
void g2(std::wstring_view sv) {
  std::wstring s;
  g2(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}g2(s);{{$}}
  g2(s.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'data' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}g2(s);{{$}}
}

// Tests for std::u16string.

void h1(const std::u16string &s) {
  h1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}h1(s);{{$}}
}
void h2(std::u16string_view sv) {
  std::u16string s;
  h2(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}h2(s);{{$}}
  h2(s.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'data' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}h2(s);{{$}}
}

// Tests for std::u32string.

void k1(const std::u32string &s) {
  k1(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}k1(s);{{$}}
}
void k2(std::u32string_view sv) {
  std::u32string s;
  k2(s.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}k2(s);{{$}}
  k2(s.data());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant call to 'data' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}k2(s);{{$}}
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

// Test for rvalue std::string.
void m1(std::string&&) {
  std::string s;

  m1(s.c_str());

  void (*m1p1)(std::string&&);
  m1p1 = m1;
  m1p1(s.c_str());

  using m1tp = void (*)(std::string &&);
  m1tp m1p2 = m1;
  m1p2(s.c_str());  
}

// Test for overloaded operator->
void it(iterator<std::string> i)
{
  std::string tmp;
  tmp = i->c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}tmp = *i;{{$}}

  // An unlikely situation and the outcome is not ideal, but at least the fix doesn't generate broken code.
  tmp = i.operator->()->c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}tmp = *i.operator->();{{$}}

  // The fix contains an unnecessary set of parentheses, but these have no effect.
  iterator<std::string> *pi = &i;
  tmp = (*pi)->c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}tmp = *(*pi);{{$}}

  // An unlikely situation, but at least the fix doesn't generate broken code.
  tmp = pi->operator->()->c_str();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}tmp = *pi->operator->();{{$}}
}

namespace PR45286 {
struct Foo {
  void func(const std::string &) {}
  void func2(std::string &&) {}
};

void bar() {
  std::string Str{"aaa"};
  Foo Foo;
  Foo.func(Str.c_str());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}Foo.func(Str);{{$}}

  // Ensure it doesn't transform Binding to r values
  Foo.func2(Str.c_str());

  // Ensure its not confused by parens
  Foo.func((Str.c_str()));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant call to 'c_str' [readability-redundant-string-cstr]
  // CHECK-FIXES: {{^  }}Foo.func((Str));{{$}}
  Foo.func2((Str.c_str()));
}
} // namespace PR45286
