// RUN: %check_clang_tidy %s performance-faster-string-find %t
// RUN: %check_clang_tidy -check-suffix=CUSTOM %s performance-faster-string-find %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {performance-faster-string-find.StringLikeClasses: \
// RUN:                '::llvm::StringRef;'}}"

namespace std {
template <typename Char>
struct basic_string {
  int find(const Char *, int = 0) const;
  int find(const Char *, int, int) const;
  int rfind(const Char *) const;
  int find_first_of(const Char *) const;
  int find_first_not_of(const Char *) const;
  int find_last_of(const Char *) const;
  int find_last_not_of(const Char *) const;
};

typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;

template <typename Char>
struct basic_string_view {
  int find(const Char *, int = 0) const;
  int find(const Char *, int, int) const;
  int rfind(const Char *) const;
  int find_first_of(const Char *) const;
  int find_first_not_of(const Char *) const;
  int find_last_of(const Char *) const;
  int find_last_not_of(const Char *) const;
};

typedef basic_string_view<char> string_view;
typedef basic_string_view<wchar_t> wstring_view;
}  // namespace std

namespace llvm {
struct StringRef {
  int find(const char *) const;
};
}  // namespace llvm

struct NotStringRef {
  int find(const char *);
};

void StringFind() {
  std::string Str;

  Str.find("a");
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: 'find' called with a string literal consisting of a single character; consider using the more effective overload accepting a character [performance-faster-string-find]
  // CHECK-FIXES: Str.find('a');

  // Works with the pos argument.
  Str.find("a", 1);
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: 'find' called with a string literal
  // CHECK-FIXES: Str.find('a', 1);

  // Doesn't work with strings smaller or larger than 1 char.
  Str.find("");
  Str.find("ab");

  // Doesn't do anything with the 3 argument overload.
  Str.find("a", 1, 1);

  // Single quotes are escaped properly
  Str.find("'");
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: 'find' called with a string literal
  // CHECK-FIXES: Str.find('\'');
  Str.find("\'");
  // CHECK-MESSAGES: [[@LINE-1]]:12: warning: 'find' called with a string literal
  // CHECK-FIXES: Str.find('\'');

  // Other methods that can also be replaced
  Str.rfind("a");
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: 'rfind' called with a string literal
  // CHECK-FIXES: Str.rfind('a');
  Str.find_first_of("a");
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: 'find_first_of' called with a string
  // CHECK-FIXES: Str.find_first_of('a');
  Str.find_first_not_of("a");
  // CHECK-MESSAGES: [[@LINE-1]]:25: warning: 'find_first_not_of' called with a
  // CHECK-FIXES: Str.find_first_not_of('a');
  Str.find_last_of("a");
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'find_last_of' called with a string
  // CHECK-FIXES: Str.find_last_of('a');
  Str.find_last_not_of("a");
  // CHECK-MESSAGES: [[@LINE-1]]:24: warning: 'find_last_not_of' called with a
  // CHECK-FIXES: Str.find_last_not_of('a');

  // std::wstring should work.
  std::wstring WStr;
  WStr.find(L"n");
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: 'find' called with a string literal
  // CHECK-FIXES: Str.find(L'n');
  // Even with unicode that fits in one wide char.
  WStr.find(L"\x3A9");
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: 'find' called with a string literal
  // CHECK-FIXES: Str.find(L'\x3A9');

  // std::string_view and std::wstring_view should work.
  std::string_view StrView;
  StrView.find("n");
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: 'find' called with a string literal
  // CHECK-FIXES: StrView.find('n');
  std::wstring_view WStrView;

  WStrView.find(L"n");
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: 'find' called with a string literal
  // CHECK-FIXES: WStrView.find(L'n');
  WStrView.find(L"\x3A9");
  // CHECK-MESSAGES: [[@LINE-1]]:17: warning: 'find' called with a string literal
  // CHECK-FIXES: WStrView.find(L'\x3A9');

  // Also with other types, but only if it was specified in the options.
  llvm::StringRef sr;
  sr.find("x");
  // CHECK-MESSAGES-CUSTOM: [[@LINE-1]]:11: warning: 'find' called with a string literal
  // CHECK-FIXES-CUSTOM: sr.find('x');
  NotStringRef nsr;
  nsr.find("x");
}


template <typename T>
int FindTemplateDependant(T value) {
  return value.find("A");
}
template <typename T>
int FindTemplateNotDependant(T pos) {
  return std::string().find("A", pos);
  // CHECK-MESSAGES: [[@LINE-1]]:29: warning: 'find' called with a string literal
  // CHECK-FIXES: return std::string().find('A', pos);
}

int FindStr() {
  return FindTemplateDependant(std::string()) + FindTemplateNotDependant(1);
}

#define STR_MACRO(str) str.find("A")
#define POS_MACRO(pos) std::string().find("A",pos)

int Macros() {
  return STR_MACRO(std::string()) + POS_MACRO(1);
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: 'find' called with a string literal
  // CHECK-MESSAGES: [[@LINE-2]]:37: warning: 'find' called with a string literal
}
