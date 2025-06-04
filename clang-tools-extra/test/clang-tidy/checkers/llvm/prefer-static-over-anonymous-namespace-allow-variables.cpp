// RUN: %check_clang_tidy %s llvm-prefer-static-over-anonymous-namespace %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     llvm-prefer-static-over-anonymous-namespace.AllowVariableDeclarations: false }, \
// RUN:   }" -- -fno-delayed-template-parsing

namespace {

void regularFunction(int param) {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'regularFunction' is declared in an anonymous namespace;

  int Variable = 42;
  auto Lambda = []() { return 42; };
  static int StaticVariable = 42;
}

int globalVariable = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'globalVariable' is declared in an anonymous namespace;

static int staticVariable = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: place static variable 'staticVariable' outside of an anonymous namespace

typedef int MyInt;
const MyInt myGlobalVariable = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'myGlobalVariable' is declared in an anonymous namespace;

template<typename T>
constexpr T Pi = T(3.1415926);
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: variable 'Pi' is declared in an anonymous namespace;

void (*funcPtr)() = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: variable 'funcPtr' is declared in an anonymous namespace;

auto lambda = []() { return 42; };
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: variable 'lambda' is declared in an anonymous namespace;

class MyClass {
  int member;
};

MyClass instance;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: variable 'instance' is declared in an anonymous namespace;

MyClass* instancePtr = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: variable 'instancePtr' is declared in an anonymous namespace;

MyClass& instanceRef = instance;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: variable 'instanceRef' is declared in an anonymous namespace;

class OtherClass{
  void method() {
    MyClass instance;
    MyClass* instancePtr = nullptr;
    MyClass& instanceRef = instance;
  }
  MyClass member;
  MyClass* memberPtr = nullptr;
  MyClass& memberRef = member;
};

} // namespace
