// RUN: %check_clang_tidy %s readability-inconsistent-declaration-parameter-name %t -- \
// RUN:   -config="{CheckOptions: [{key: readability-inconsistent-declaration-parameter-name.IgnoreMacros, value: 0}]}" \
// RUN:   -- -std=c++11

#define MACRO() \
  void f(int x);

struct S {
  MACRO();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: function 'S::f' has a definition with different parameter names
};

void S::f(int y) {
}

//////////////////////////////////////////////////////

#define DECLARE_FUNCTION_WITH_PARAM_NAME(function_name, param_name) \
  void function_name(int param_name)

// CHECK-MESSAGES: :[[@LINE+1]]:34: warning: function 'macroFunction' has 1 other declaration with different parameter names [readability-inconsistent-declaration-parameter-name]
DECLARE_FUNCTION_WITH_PARAM_NAME(macroFunction, a);
// CHECK-MESSAGES: :[[@LINE+2]]:34: note: the 1st inconsistent declaration seen here
// CHECK-MESSAGES: :[[@LINE+1]]:34: note: differing parameters are named here: ('b'), in the other declaration: ('a')
DECLARE_FUNCTION_WITH_PARAM_NAME(macroFunction, b);
