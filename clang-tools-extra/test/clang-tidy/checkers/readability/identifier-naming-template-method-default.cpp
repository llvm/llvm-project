// RUN: %check_clang_tidy %s readability-identifier-naming %t -std=c++20-or-later \
// RUN:   --config='{CheckOptions: { \
// RUN:     readability-identifier-naming.DefaultCase: lower_case, \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.FunctionCase: camelBack, \
// RUN:     readability-identifier-naming.MethodCase: camelBack, \
// RUN:  }}'

class Foo {
public:
  template <typename t>
  void doStuff() {}

  template <typename t>
  void DoStuff() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for method 'DoStuff' [readability-identifier-naming]
  // CHECK-FIXES: void doStuff() {}
};

template <typename t>
void freeFunction() {}

template <typename t>
void FreeFunction() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for function 'FreeFunction' [readability-identifier-naming]
// CHECK-FIXES: void freeFunction() {}

int BadGlobal = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for identifier 'BadGlobal' [readability-identifier-naming]
// CHECK-FIXES: int bad_global = 0;
