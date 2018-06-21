// RUN: %check_clang_tidy %s misc-unconventional-assign-operator %t -- -- -std=c++17 -fno-delayed-template-parsing

struct BadModifier {
  BadModifier& operator=(const BadModifier&) const;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: operator=() should not be marked 'const'
};

struct PR35468 {
  template<typename T> auto &operator=(const T &) {
    return *this;
  }
};
