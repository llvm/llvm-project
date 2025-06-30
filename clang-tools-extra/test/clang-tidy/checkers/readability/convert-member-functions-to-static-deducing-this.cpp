// RUN: %check_clang_tidy -std=c++23-or-later %s readability-convert-member-functions-to-static %t

namespace std{
  class string {};
  void println(const char *format, const std::string &str) {}
}

struct Hello {
  std::string str_;

  void ByValueSelf(this Hello self) { std::println("Hello, {0}!", self.str_); }

  void ByLRefSelf(this Hello &self) { std::println("Hello, {0}!", self.str_); }

  void ByRRefSelf(this Hello&& self) {}

  template<typename Self> void ByForwardRefSelf(this Self&& self) {}

  void MultiParam(this Hello &self, int a, double b) {}

  void UnnamedExplicitObjectParam(this Hello &) {}
};
