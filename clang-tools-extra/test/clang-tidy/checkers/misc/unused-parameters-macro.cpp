// RUN: %check_clang_tidy %s misc-unused-parameters %t -- \
// RUN:   -config="{CheckOptions: {misc-unused-parameters.IgnoreMacroParameters: true, \
// RUN:             misc-unused-parameters.StrictMode: true}}" --

// Parameters declared inside a macro expansion should not be reported when
// IgnoreMacroParameters is enabled.

#define HANDLER(RetType, ParamType, ParamName) \
  RetType handler_##ParamName(ParamType ParamName) { return 0; }

HANDLER(int, int, unused_from_macro)
// No warning: parameter comes from macro expansion.

#define WRAP_PARAM(type) type macro_arg
void functionWithMacroParam(WRAP_PARAM(int)) {}
// No warning: parameter name comes from macro expansion.

// Regular unused parameter should still be reported.
void normalFunction(int unused_normal) {}
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: parameter 'unused_normal' is unused [misc-unused-parameters]
// CHECK-FIXES: void normalFunction(int  /*unused_normal*/) {}
