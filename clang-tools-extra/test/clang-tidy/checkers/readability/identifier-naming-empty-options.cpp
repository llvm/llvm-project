// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.GlobalConstantPrefix: "", \
// RUN:     readability-identifier-naming.GlobalVariablePrefix: g_ }}'

int BadGlobalVariable;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'BadGlobalVariable' [readability-identifier-naming]
// CHECK-FIXES: int g_BadGlobalVariable;
int g_GoodGlobalVariable;

const int GoodGlobalConstant = 0;
const int g_IgnoreGlobalConstant = 0;
