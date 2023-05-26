// RUN: %check_clang_tidy %s bugprone-assert-side-effect %t -- -config="{CheckOptions: [{key: bugprone-assert-side-effect.CheckFunctionCalls, value: true}, {key: bugprone-assert-side-effect.AssertMacros, value: 'assert,assert2,my_assert,convoluted_assert,msvc_assert'}, {key: bugprone-assert-side-effect.IgnoredFunctions, value: 'MyClass::badButIgnoredFunc'}]}" -- -fexceptions -I %S/Inputs/assert-side-effect
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

  return 0;
}
