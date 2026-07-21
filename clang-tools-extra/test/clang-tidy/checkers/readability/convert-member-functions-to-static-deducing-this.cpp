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

class BaseOverloadedAutoLambdaTest {
public:
  void BaseOverloadedMethod();
  void BaseOverloadedMethod(int a) { };
};


class OverloadedAutoLambdaTest : public BaseOverloadedAutoLambdaTest {
public:
  void CallFunVar();
  // CHECK-FIXES: static void CallFunVar();
  void LambdaUsesThis1(int a);
  void LambdaUsesThis2(int a);
  void LambdaUsesThis3(int a);
  void LambdaUsesThis4();
  void LambdaUsesThis5();

  void LambdaNoThis1(int a);
  // CHECK-FIXES: static void LambdaNoThis1(int a);
  void LambdaNoThis2(int a);
  // CHECK-FIXES: static void LambdaNoThis2(int a);
  void LambdaNoThis3(int a);
  // CHECK-FIXES: static void LambdaNoThis3(int a);
  void LambdaNoThis4(int a);
  // CHECK-FIXES: static void LambdaNoThis4(int a);
  void LambdaNoThis5(int a);
  // CHECK-FIXES: static void LambdaNoThis5(int a);

  void OverloadedMethod();
  void OverloadedMethod(int a) { };

  template <typename T>
  void TemplatedOverloadedMethod(T);
  void TemplatedOverloadedMethod(int);
};

void OverloadedAutoLambdaTest::CallFunVar() {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: method 'CallFunVar' can be made static [readability-convert-member-functions-to-static]
  auto fun = [&](auto a) {
    var(a);
  };
}

void OverloadedAutoLambdaTest::LambdaUsesThis1(int a) {
  auto fun = [&](auto b) {
    OverloadedMethod(b);
  };
}

void OverloadedAutoLambdaTest::LambdaUsesThis2(int a) {
  auto fun = [&](auto b) {
    this->OverloadedMethod(b);
  };
}

void OverloadedAutoLambdaTest::LambdaNoThis1(int a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: method 'LambdaNoThis1' can be made static [readability-convert-member-functions-to-static]
  OverloadedAutoLambdaTest f;
  auto fun = [&](auto b) {
    f.OverloadedMethod(b);
  };
}

void OverloadedAutoLambdaTest::LambdaUsesThis3(int a) {
  OverloadedAutoLambdaTest g;
  auto fun1 = [&](auto a) {
    auto fun2 = [&](auto b) {
      OverloadedMethod(b);
    };
    fun2(a);
  };
}

void OverloadedAutoLambdaTest::LambdaNoThis2(int a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: method 'LambdaNoThis2' can be made static [readability-convert-member-functions-to-static]
  OverloadedAutoLambdaTest g;
  auto fun1 = [&](auto a) {
    auto fun2 = [&](auto b) {
      g.OverloadedMethod(b);
    };
    fun2(a);
  };
}

void OverloadedAutoLambdaTest::LambdaUsesThis4() {
  auto fun = [&](auto a) {
    OverloadedMethod(a);
  };
}

void OverloadedAutoLambdaTest::LambdaNoThis3(int a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: method 'LambdaNoThis3' can be made static [readability-convert-member-functions-to-static]
  OverloadedAutoLambdaTest f;
  auto fun = [&](auto b) {
    f.OverloadedMethod(b);
  };
}

void OverloadedAutoLambdaTest::LambdaNoThis4(int a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: method 'LambdaNoThis4' can be made static [readability-convert-member-functions-to-static]
  BaseOverloadedAutoLambdaTest f;
  auto fun = [&](auto b) {
    f.BaseOverloadedMethod(b);
  };
}

void OverloadedAutoLambdaTest::LambdaNoThis5(int a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: method 'LambdaNoThis5' can be made static [readability-convert-member-functions-to-static]
  OverloadedAutoLambdaTest f;
  auto fun = [&](auto b) {
    f.BaseOverloadedMethod(b);
  };
}

void OverloadedAutoLambdaTest::LambdaUsesThis5() {
  auto fun = [&](auto b) {
    TemplatedOverloadedMethod(b);
  };
}

void Var(int a) { }
