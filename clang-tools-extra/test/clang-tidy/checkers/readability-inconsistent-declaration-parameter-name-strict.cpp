// RUN: %check_clang_tidy %s readability-inconsistent-declaration-parameter-name %t -- \
// RUN:   -config="{CheckOptions: [{key: readability-inconsistent-declaration-parameter-name.Strict, value: true}]}"

void inconsistentFunction(int a, int b, int c);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'inconsistentFunction' has 1 other declaration with different parameter names
void inconsistentFunction(int prefixA, int b, int cSuffix);
// CHECK-MESSAGES: :[[@LINE-1]]:6: note: the 1st inconsistent declaration seen here
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: differing parameters are named here: ('prefixA', 'cSuffix'), in the other declaration: ('a', 'c')
void inconsistentFunction(int a, int b, int c);
void inconsistentFunction(int /*c*/, int /*c*/, int /*c*/);
