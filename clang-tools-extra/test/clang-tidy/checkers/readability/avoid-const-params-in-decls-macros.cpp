// RUN: %check_clang_tidy %s readability-avoid-const-params-in-decls %t -- \
// RUN:   -config="{CheckOptions: {readability-avoid-const-params-in-decls.IgnoreMacros: false}}"

// Regression tests involving macros
#define CONCAT(a, b) a##b
void ConstNotVisible(CONCAT(cons, t) int i);
// CHECK-MESSAGES: :[[@LINE-1]]:22: warning: parameter 'i'
// We warn, but we can't give a fix
// CHECK-FIXES: void ConstNotVisible(CONCAT(cons, t) int i);

#define CONST_INT_PARAM const int i
void ConstInMacro(CONST_INT_PARAM);
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: parameter 'i'
// We warn, but we can't give a fix
// CHECK-FIXES: void ConstInMacro(CONST_INT_PARAM);

#define DECLARE_FUNCTION_WITH_ARG(x) struct InsideMacro{ x }
DECLARE_FUNCTION_WITH_ARG(
    void member_function(const int i);
);
// CHECK-MESSAGES: :[[@LINE-2]]:26: warning: parameter 'i'
// CHECK-FIXES: void member_function(int i);
