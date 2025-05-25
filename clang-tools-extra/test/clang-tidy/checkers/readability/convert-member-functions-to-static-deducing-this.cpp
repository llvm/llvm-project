// RUN: %check_clang_tidy -std=c++23 %s readability-convert-member-functions-to-static %t

namespace std{
  class string {};
  void println(const char *format, const std::string &str) {}
}

namespace PR141381 {
struct Hello {
  std::string str_;

  void hello(this Hello &self) { std::println("Hello, {0}!", self.str_); }
};
}
