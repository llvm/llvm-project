// RUN: %check_clang_tidy -std=c++23-or-later %s readability-convert-member-functions-to-static %t
#include <string>

namespace std{
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


class OverloadedUnresolvedWithAutoLambda {
public:
  void CallsFunctionVar();
  void CallsOverloadedMethodWithArg(int a);
  void OverloadedMethod();
  void OverloadedMethod(int a) { };
};

void OverloadedUnresolvedWithAutoLambda::CallsFunctionVar() {
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: method 'CallsFunctionVar' can be made static [readability-convert-member-functions-to-static]
  auto fun = [&](auto a) {
    var(a);
  };
}

void OverloadedUnresolvedWithAutoLambda::CallsOverloadedMethodWithArg(int a) {
  auto fun = [&](auto b) {
    OverloadedMethod(b);
  };
}

void Var(int a) { }
