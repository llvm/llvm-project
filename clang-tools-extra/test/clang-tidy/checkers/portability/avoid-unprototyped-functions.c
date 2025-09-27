// RUN: %check_clang_tidy %s portability-avoid-unprototyped-functions %t
struct S {
  int (*UnprototypedFunctionPtrField)();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid unprototyped functions in type specifiers; explicitly add a 'void' parameter if the function takes no arguments
};

void unprototypedFunction();
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid unprototyped function declarations; explicitly spell out a single 'void' parameter if the function takes no argument

void unprototypedParamter(int (*UnprototypedFunctionPtrParameter)());
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning: avoid unprototyped functions in type specifiers; explicitly add a 'void' parameter if the function takes no arguments

void unprototypedVariable() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid unprototyped function declarations; explicitly spell out a single 'void' parameter if the function takes no argument

  int (*UnprototypedFunctionPtrVariable)();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: avoid unprototyped functions in type specifiers; explicitly add a 'void' parameter if the function takes no arguments
}

typedef int (*UnprototypedFunctionPtrArray[13])();
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: avoid unprototyped functions in typedef; explicitly add a 'void' parameter if the function takes no arguments

void unprototypedTypeAliasParameter(UnprototypedFunctionPtrArray);

#define ANYARG

void unprototypedMacroParameter(ANYARG);
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: avoid unprototyped function declarations; explicitly spell out a single 'void' parameter if the function takes no argument

void functionWithNoArguments(void);

int main();
