// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-unsafe-to-allow-exceptions %t -- -- -fexceptions
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffix=,CUSTOMSWAP %s bugprone-unsafe-to-allow-exceptions %t -- \
// RUN:     -config="{CheckOptions: { \
// RUN:         bugprone-unsafe-to-allow-exceptions.CheckedSwapFunctions: 'swap;iter_swap;iter_move;swap1', \
// RUN:     }}" \
// RUN: -- -fexceptions

struct may_throw {
  may_throw(may_throw&&) noexcept(false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'may_throw' should not throw exceptions but it is still marked as potentially throwing [bugprone-unsafe-to-allow-exceptions]
  }
  may_throw& operator=(may_throw&&) noexcept(false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: function 'operator=' should not throw exceptions but it is still marked as potentially throwing
  }
  ~may_throw() noexcept(false) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function '~may_throw' should not throw exceptions but it is still marked as potentially throwing
  }

  void f() noexcept(false) {
  }
};

struct no_throw {
  no_throw(no_throw&&) throw() {
  }
  no_throw& operator=(no_throw&&) noexcept(true) {
  }
  ~no_throw() noexcept(true) {
  }
};

int main() noexcept(false) {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'main' should not throw exceptions but it is still marked as potentially throwing
  return 0;
}

void swap(int&, int&) noexcept(false) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'swap' should not throw exceptions but it is still marked as potentially throwing
}

void iter_swap(int&, int&) noexcept(false) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'iter_swap' should not throw exceptions but it is still marked as potentially throwing
}

void iter_move(int&) noexcept(false) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'iter_move' should not throw exceptions but it is still marked as potentially throwing
}

void swap(double&, double&) {
}

void swap1(long&) noexcept(false) {
  // CHECK-MESSAGES-CUSTOMSWAP: :[[@LINE-1]]:6: warning: function 'swap1' should not throw exceptions but it is still marked as potentially throwing
}
