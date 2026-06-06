// RUN: %clang_analyze_cc1 %s -verify -analyzer-checker=core

#include "Inputs/system-header-simulator-cxx.h"


namespace GH94193 {
template<typename T> class optional {
  union {
    char x;
    T uvalue;
  };
  bool holds_value = false;
public:
  optional() = default;
  optional(const optional&) = delete;
  optional(optional&&) = delete;
  template <typename U = T> explicit optional(U&& value) : holds_value(true) {
    new (static_cast<void*>(std::addressof(uvalue))) T(std::forward<U>(value));
  }
  optional& operator=(const optional&) = delete;
  optional& operator=(optional&&) = delete;
  explicit operator bool() const {
    return holds_value;
  }
  T& unwrap() & {
    return uvalue; // no-warning: returns a valid value
  }
};

int top1(int x) {
  optional<int> opt{x}; // note: Ctor was inlined.
  return opt.unwrap();  // no-warning: returns a valid value
}

std::string *top2() {
  std::string a = "123";
  // expected-warning@+2 {{address of stack memory associated with local variable 'a' returned}} diagnosed by -Wreturn-stack-address
  // expected-warning@+1 {{Address of stack memory associated with local variable 'a' returned to caller [core.StackAddressEscape]}}
  return std::addressof(a);
}
} // namespace GH94193
