// RUN: %check_clang_tidy %s readability-identifier-naming %t -std=c++20-or-later \
// RUN:   --config='{CheckOptions: { \
// RUN:     readability-identifier-naming.DefaultCase: lower_case, \
// RUN:     readability-identifier-naming.ClassCase: CamelCase, \
// RUN:     readability-identifier-naming.FunctionCase: camelBack, \
// RUN:     readability-identifier-naming.GlobalVariableCase: camelBack, \
// RUN:     readability-identifier-naming.MethodCase: camelBack, \
// RUN:     readability-identifier-naming.TypeAliasCase: lower_case, \
// RUN:     readability-identifier-naming.TypeAliasSuffix: _t, \
// RUN:  }}'

#define BadMacro
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for identifier 'BadMacro' [readability-identifier-naming]
// CHECK-FIXES: #define bad_macro

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

template <typename t>
class GoodClass {};

template <typename t>
class badClass {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'badClass' [readability-identifier-naming]
// CHECK-FIXES: class BadClass {};

template <typename t>
int goodVariable = 0;

template <typename t>
int BadVariable = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'BadVariable' [readability-identifier-naming]
// CHECK-FIXES: int badVariable = 0;

template <typename t>
using good_alias_t = int;

template <typename t>
using BadAlias = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for type alias 'BadAlias' [readability-identifier-naming]
// CHECK-FIXES: using bad_alias_t = int;

int BadGlobal = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'BadGlobal' [readability-identifier-naming]
// CHECK-FIXES: int badGlobal = 0;
