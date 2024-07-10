// RUN: %check_clang_tidy %s bugprone-assert-side-effect %t -- -config="{CheckOptions: {bugprone-assert-side-effect.CheckFunctionCalls: true, bugprone-assert-side-effect.AssertMacros: 'assert,assert2,my_assert,convoluted_assert,msvc_assert', bugprone-assert-side-effect.IgnoredFunctions: 'MyClass::badButIgnoredFunc'}}" -- -fexceptions -I %S/Inputs/assert-side-effect
#include <assert.h>

bool badButIgnoredFunc(int a, int b) { return a * b > 0; }

class MyClass {
public:
  bool badFunc(int a, int b) { return a * b > 0; }
  bool badButIgnoredFunc(int a, int b) { return a * b > 0; }
  bool goodFunc(int a, int b) const { return a * b > 0; }

  MyClass &operator=(const MyClass &rhs) { return *this; }

  int operator-() { return 1; }

  operator bool() const { return true; }

  void operator delete(void *p) {}
};

class SomeoneElseClass {
public:
  bool badButIgnoredFunc(int a, int b) { return a * b > 0; }
};

bool freeFunction() {
  return true;
}

int main() {

  int X = 0;
  bool B = false;
  assert(X == 1);

  assert(X = 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds [bugprone-assert-side-effect]
  my_assert(X = 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in my_assert() condition discarded in release builds
  convoluted_assert(X = 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in convoluted_assert() condition discarded in release builds
  not_my_assert(X = 1);

  assert(++X);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds
  assert(!B);

  assert(B || true);

  assert(freeFunction());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds

  MyClass mc;
  SomeoneElseClass sec;
  assert(mc.badFunc(0, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds
  assert(mc.badButIgnoredFunc(0, 1));
  // badButIgnoredFunc is not ignored as only class members are ignored by the config
  assert(badButIgnoredFunc(0, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds
  // sec.badButIgnoredFunc is not ignored as only MyClass members are ignored by the config
  assert(sec.badButIgnoredFunc(0, 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds
  assert(mc.goodFunc(0, 1));

  MyClass mc2;
  assert(mc2 = mc);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds

  assert(-mc > 0);

  MyClass *mcp;
  assert(mcp = new MyClass);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds

  assert((delete mcp, false));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds

  assert((throw 1, false));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds

  assert2(1 == 2 - 1);

  msvc_assert(mc2 = mc);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in msvc_assert() condition discarded in release builds

  struct OperatorTest {
    int operator<<(int i) const { return i; }
    int operator<<(int i) { return i; }
    int operator+=(int i) const { return i; }
    int operator+=(int i) { return i; }
  };

  const OperatorTest const_instance;
  assert(const_instance << 1);
  assert(const_instance += 1);

  OperatorTest non_const_instance;
  assert(static_cast<const OperatorTest>(non_const_instance) << 1);
  assert(static_cast<const OperatorTest>(non_const_instance) += 1);
  assert(non_const_instance << 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds
  assert(non_const_instance += 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds

  assert(5<<1);
  assert(5>>1);

  return 0;
}

namespace parameter_anaylysis {

struct S {
  bool value(int) const;
  bool leftValueRef(int &) const;
  bool constRef(int const &) const;
  bool rightValueRef(int &&) const;
};

void foo() {
  S s{};
  int i = 0;
  assert(s.value(0));
  assert(s.value(i));
  assert(s.leftValueRef(i));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: side effect in assert() condition discarded in release builds
  assert(s.constRef(0));
  assert(s.constRef(i));
  assert(s.rightValueRef(0));
  assert(s.rightValueRef(static_cast<int &&>(i)));
}

} // namespace parameter_anaylysis
