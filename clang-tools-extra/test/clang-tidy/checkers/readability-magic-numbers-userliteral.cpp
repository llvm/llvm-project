// RUN: %check_clang_tidy -std=c++14-or-later %s readability-magic-numbers %t --

namespace std {
  class string {};
  using size_t = decltype(sizeof(int));
  string operator ""s(const char *, std::size_t);
  int operator "" s(unsigned long long);
}

void UserDefinedLiteral() {
  using std::operator ""s;
  "Hello World"s;
  const int i = 3600s;
  int j = 3600s;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 3600s is a magic number; consider replacing it with a named constant [readability-magic-numbers]
}
