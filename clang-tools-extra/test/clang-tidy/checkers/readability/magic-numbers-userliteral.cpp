// RUN: %check_clang_tidy -check-suffixes=UDL-ALLOWED -std=c++14-or-later %s readability-magic-numbers %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-magic-numbers.IgnoreUserDefinedLiterals: false}}' \
// RUN: --
// RUN: %check_clang_tidy -check-suffixes=UDL-IGNORED -std=c++14-or-later %s readability-magic-numbers %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-magic-numbers.IgnoreUserDefinedLiterals: true}}' \
// RUN: --

namespace std {
  class string {};
  using size_t = decltype(sizeof(int));
  string operator ""s(const char *, std::size_t);
  int operator "" s(unsigned long long);
  float operator "" s(long double);
}

void UserDefinedLiteral() {
  using std::operator ""s;
  "Hello World"s;
  const int i = 3600s;
  int j = 3600s;
  // CHECK-MESSAGES-UDL-ALLOWED: :[[@LINE-1]]:11: warning: 3600s is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  // CHECK-MESSAGES-UDL-IGNORED-NOT: :[[@LINE-2]]:11: warning: 3600s is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  float k = 3600.0s;
  // CHECK-MESSAGES-UDL-ALLOWED: :[[@LINE-1]]:13: warning: 3600.0s is a magic number; consider replacing it with a named constant [readability-magic-numbers]
  // CHECK-MESSAGES-UDL-IGNORED-NOT: :[[@LINE-1]]:13: warning: 3600.0s is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}
