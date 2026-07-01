// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.FunctionCase: camelBack, \
// RUN:     readability-identifier-naming.ParameterCase: camelBack, \
// RUN:     readability-identifier-naming.VariableCase: camelBack \
// RUN:   }}'

#define WRAP(E) E

WRAP(int foo(int v) { return v; })

void testLambdaInMacroArgument() {
  WRAP([](int var) {
    return var;
  }(1));
}

int badFunction(int BadParam) {
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for parameter 'BadParam' [readability-identifier-naming]
  // CHECK-FIXES: int badFunction(int badParam) {
  return BadParam;
  // CHECK-FIXES: return badParam;
}
