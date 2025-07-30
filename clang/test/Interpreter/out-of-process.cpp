// REQUIRES: host-supports-jit,  host-supports-out-of-process-jit, x86_64-linux

// RUN: cat %s | clang-repl -oop-executor -orc-runtime | FileCheck %s

extern "C" int printf(const char *, ...);

int intVar = 0;
double doubleVar = 3.14;
%undo
double doubleVar = 2.71;

auto r1 = printf("intVar = %d\n", intVar);
// CHECK: intVar = 0
auto r2 = printf("doubleVar = %.2f\n", doubleVar);
// CHECK: doubleVar = 2.71

// Test redefinition with inline and static functions.
int add(int a, int b, int c) { return a + b + c; }
%undo  // Revert to the initial version of add
inline int add(int a, int b) { return a + b; }

auto r3 = printf("add(1, 2) = %d\n", add(1, 2));
// CHECK-NEXT: add(1, 2) = 3

// Test inline and lambda functions with variations.
inline int square(int x) { return x * x; }
auto lambdaSquare = [](int x) { return x * x; };
auto lambdaMult = [](int a, int b) { return a * b; };

auto r4 = printf("square(4) = %d\n", square(4));
// CHECK-NEXT: square(4) = 16
auto lambda_r1 = printf("lambdaSquare(5) = %d\n", lambdaSquare(5));
// CHECK-NEXT: lambdaSquare(5) = 25
auto lambda_r2 = printf("lambdaMult(2, 3) = %d\n", lambdaMult(2, 3));
// CHECK-NEXT: lambdaMult(2, 3) = 6

%undo  // Undo previous lambda assignments
auto lambda_r3 = lambdaMult(3, 4);  // Should fail or revert to the original lambda

// Test weak and strong symbol linkage.
int __attribute__((weak)) weakFunc() { return 42; }
int strongFunc() { return 100; }
%undo  // Revert the weak function

auto r5 = printf("weakFunc() = %d\n", weakFunc());
// CHECK: weakFunc() = 42
auto r6 = printf("strongFunc() = %d\n", strongFunc());
// CHECK-NEXT: strongFunc() = 100

// Weak variable linkage with different types.
int varA = 20;
static __typeof(varA) weakVarA __attribute__((__weakref__("varA")));
char charVar = 'c';
static __typeof(charVar) weakCharVar __attribute__((__weakref__("charVar")));
auto r7 = printf("weakVarA = %d\n", weakVarA);
// CHECK: weakVarA = 20
auto r8 = printf("weakCharVar = %c\n", weakCharVar);
// CHECK-NEXT: weakCharVar = c

// Test complex lambdas with captures.
int captureVar = 5;
auto captureLambda = [](int x) { return x + captureVar; };
int result1 = captureLambda(10);
%undo  // Undo capture lambda

auto r9 = printf("captureLambda(10) = %d\n", result1);
// CHECK: captureLambda(10) = 15

// Multiline statement test with arithmetic operations.
int sum = \
  5 + \
  10;
int prod = sum * 2;
auto r10 = printf("sum = %d, prod = %d\n", sum, prod);
// CHECK: sum = 15, prod = 30

// Test multiline functions and macro behavior.
#define MULTIPLY(a, b) ((a) * (b))

int complexFunc(int x) \
{ \
  return MULTIPLY(x, 2) + x; \
}

auto r11 = printf("complexFunc(5) = %d\n", complexFunc(5));
// CHECK: complexFunc(5) = 15

%quit