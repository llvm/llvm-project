// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-unsafe-to-allow-exceptions %t -- \
// RUN:     -config="{CheckOptions: { \
// RUN:         bugprone-unsafe-to-allow-exceptions.CheckedSwapFunctions: 'swap', \
// RUN:     }}" \
// RUN: -- -fexceptions

class Exception {};

struct may_throw {
  may_throw(may_throw&&) throw(int) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'may_throw' should not throw exceptions but it is still marked as potentially throwing [bugprone-unsafe-to-allow-exceptions]
  }
  may_throw& operator=(may_throw&&) throw(Exception) {
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: function 'operator=' should not throw exceptions but it is still marked as potentially throwing
  }
  ~may_throw() throw(char, Exception) {
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

int main() throw(char) {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: function 'main' should not throw exceptions but it is still marked as potentially throwing
  return 0;
}

void swap(no_throw&, no_throw&) throw(bool) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'swap' should not throw exceptions but it is still marked as potentially throwing
}

void iter_swap(int&, int&) throw(bool) {
}

void iter_move(int&) throw(bool) {
}

void swap(double&, double&) {
}
