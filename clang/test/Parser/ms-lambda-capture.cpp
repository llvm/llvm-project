// RUN: %clang_cc1 -std=c++23 -fms-extensions -fsyntax-only -verify %s
// expected-no-diagnostics

struct Wrapper {
  template <typename F> Wrapper(F) {}
};

void omitted_parameter_list() {
  Wrapper a([] mutable {});
  Wrapper b([] constexpr {});
  Wrapper c([] consteval {});
  Wrapper d([] static {});
  Wrapper e([] mutable constexpr noexcept -> int { return 1; });
  Wrapper f([] mutable [[]] {});
}

void named_capture() {
  int x = 0;
  Wrapper a([x] mutable { return x; });
}

void microsoft_attribute_control() {
  Wrapper declaration([propget] int);
}
