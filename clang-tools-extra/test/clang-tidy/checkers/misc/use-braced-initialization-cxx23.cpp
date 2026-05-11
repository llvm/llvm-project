// RUN: %check_clang_tidy -std=c++23-or-later %s misc-use-braced-initialization %t

struct Simple {
  Simple(int);
};

void auto_functional_cast() {
  auto x = auto(1);
}

void auto_functional_cast_class() {
  auto x = auto(Simple(1));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use braced initialization instead of parenthesized initialization [misc-use-braced-initialization]
  // CHECK-FIXES: auto x = auto(Simple{1});
}
