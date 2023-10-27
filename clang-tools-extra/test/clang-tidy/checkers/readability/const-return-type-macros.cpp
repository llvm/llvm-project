// RUN: %check_clang_tidy -std=c++14-or-later %s readability-const-return-type %t -- \
// RUN:   -config="{CheckOptions: {readability-const-return-type.IgnoreMacros: false}}"

//  p# = positive test
//  n# = negative test

// Regression tests involving macros
#define CONCAT(a, b) a##b
CONCAT(cons, t) int p22(){}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu
// We warn, but we can't give a fix

#define CONSTINT const int
CONSTINT p23() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu

#define CONST const
CONST int p24() {}
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu

#define CREATE_FUNCTION()                    \
const int p_inside_macro() { \
  return 1; \
}
CREATE_FUNCTION();
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: return type 'const int' is 'const'-qu
// We warn, but we can't give a fix
