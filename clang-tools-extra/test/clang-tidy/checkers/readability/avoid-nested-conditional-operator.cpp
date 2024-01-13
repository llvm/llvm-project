// RUN: %check_clang_tidy %s readability-avoid-nested-conditional-operator %t

int NestInConditional = (true ? true : false) ? 1 : 2;
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: don't use nested conditional operator
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: outside conditional operator here

int NestInTrue = true ? (true ? 1 : 2) : 2;
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: don't use nested conditional operator
// CHECK-MESSAGES: :[[@LINE-2]]:18: note: outside conditional operator here

int NestInFalse = true ? 1 : true ? 1 : 2;
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: don't use nested conditional operator
// CHECK-MESSAGES: :[[@LINE-2]]:19: note: outside conditional operator here
int NestInFalse2 = true ? 1 : (true ? 1 : 2);
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: don't use nested conditional operator
// CHECK-MESSAGES: :[[@LINE-2]]:20: note: outside conditional operator here

int NestWithParensis = true ? 1 : ((((true ? 1 : 2))));
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: don't use nested conditional operator
// CHECK-MESSAGES: :[[@LINE-2]]:24: note: outside conditional operator here

#define CONDITIONAL_EXPR (true ? 1 : 2)
int NestWithMacro = true ? CONDITIONAL_EXPR : 2;
